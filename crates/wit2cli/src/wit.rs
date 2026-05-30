//! Extract a [`LibrarySurface`] from a Wasm component's WIT.
//!
//! The surface is a flat IR over the supported subset of WIT types
//! that `component run` can map onto a `clap` CLI. Resources are
//! rejected because they cannot be sensibly represented on the
//! command line.
//!
//! ## Implementation
//!
//! The walk is built directly on `wasmparser::Validator`: we stream
//! the component bytes through the validator, record the root
//! component's export names as we encounter the
//! `ComponentExportSection`, then look each one up post-validation
//! via [`TypesRef::component_entity_type_of_export`]. This avoids the
//! `wit_component`/`wit_parser` round-trip and keeps the dependency
//! surface honest: every WIT-shape decision is grounded in the
//! validated type tables that wasmtime itself uses at runtime.
//!
//! See `_/wasmtime/crates/environ/src/component/translate.rs` for
//! the canonical reference of this validator-driven pattern.

use std::collections::HashMap;

use wasmparser::component_types::{
    ComponentDefinedType, ComponentDefinedTypeId, ComponentEntityType, ComponentFuncType,
    ComponentFuncTypeId, ComponentInstanceType, ComponentInstanceTypeId, ComponentValType,
};
use wasmparser::types::TypesRef;
use wasmparser::{
    Encoding, Parser, Payload, PrimitiveValType, ValidPayload, Validator, WasmFeatures,
};

/// Logical path to a single exported function on a component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncPath {
    /// `Some(name)` when the function lives inside a nested
    /// interface export; `None` for free world-level exports.
    pub interface: Option<String>,
    /// The function's name as declared in the WIT.
    pub func: String,
}

/// Local IR mirroring the supported subset of WIT types.
///
/// `WitTy::Record` and `WitTy::Variant` preserve WIT declaration
/// order, which is mandatory: wasmtime's runtime checks record fields
/// by position and name (see
/// `wasmtime/src/runtime/component/values.rs`), so we have to emit
/// them in the order they were declared.
// r[impl run.library-args]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitTy {
    /// `bool`
    Bool,
    /// `s8`
    S8,
    /// `s16`
    S16,
    /// `s32`
    S32,
    /// `s64`
    S64,
    /// `u8`
    U8,
    /// `u16`
    U16,
    /// `u32`
    U32,
    /// `u64`
    U64,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `char`
    Char,
    /// `string`
    String,
    /// `list<T>`
    List(Box<WitTy>),
    /// `option<T>`
    Option(Box<WitTy>),
    /// `result<T, E>` (either side may be absent).
    Result {
        /// The success-payload type, or `None` for `result<_, E>`.
        ok: Option<Box<WitTy>>,
        /// The error-payload type, or `None` for `result<T, _>`.
        err: Option<Box<WitTy>>,
    },
    /// `record { name: type, ... }` — fields preserved in WIT
    /// declaration order.
    Record(Vec<(String, WitTy)>),
    /// `variant { case, case(payload), ... }`.
    Variant(Vec<(String, Option<Box<WitTy>>)>),
    /// `enum { case-a, case-b, ... }`.
    Enum(Vec<String>),
    /// `flags { flag-a, flag-b, ... }`.
    Flags(Vec<String>),
    /// `tuple<T1, T2, ...>`.
    Tuple(Vec<WitTy>),
}

/// A single function parameter.
#[derive(Debug, Clone)]
pub struct ParamDecl {
    /// Parameter name as declared in the WIT.
    pub name: String,
    /// Parameter type.
    pub ty: WitTy,
}

/// A single function result. Currently unnamed.
#[derive(Debug, Clone)]
pub struct ResultDecl {
    /// Type of the result. Used by the wire-up to validate the
    /// number of returned values matches the declared signature
    /// and to drive future type-aware error messages.
    pub ty: WitTy,
}

/// A single exported function.
#[derive(Debug, Clone)]
pub struct FuncDecl {
    /// Function name as declared in the WIT.
    pub name: String,
    /// Doc-comment, used as the clap `about` text.
    pub doc: Option<String>,
    /// Parameters in declaration order.
    pub params: Vec<ParamDecl>,
    /// Function results, used to populate
    /// [`crate::Invocation::expected_results`] for runtime sanity
    /// checks.
    pub results: Vec<ResultDecl>,
}

/// A top-level item in the library surface.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LibraryItem {
    /// Free function exported at the world level.
    Func(FuncDecl),
    /// An exported interface containing one or more functions.
    Interface {
        /// Short, user-facing name (e.g. `math`).
        name: String,
        /// Fully-qualified export name used by wasmtime
        /// (`namespace:pkg/iface@version`). May equal `name` when the
        /// interface was declared inline at the world level.
        export_name: String,
        /// Doc-comment declared on the interface, if any.
        doc: Option<String>,
        /// Functions exported by the interface, in WIT order.
        funcs: Vec<FuncDecl>,
    },
}

/// The full set of dynamically-dispatchable exports of a component.
#[derive(Debug, Clone)]
#[must_use]
pub struct LibrarySurface {
    /// Top-level items (functions and interfaces).
    pub items: Vec<LibraryItem>,
}

/// Errors raised when we cannot extract a usable surface.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LibraryExtractError {
    /// The component bytes could not be decoded as a WIT-bearing
    /// component.
    #[error("failed to decode component WIT: {0}")]
    Decode(String),
    /// The component is a WIT package, not a compiled component.
    #[error("input is a WIT package, not a compiled component")]
    NotAComponent,
    // r[impl run.library-resources-rejected]
    /// The component exports a resource type, which cannot be
    /// expressed as a CLI argument.
    #[error("resource type `{name}` is not supported by `component run`")]
    Resource {
        /// Name of the resource type (or interface) that triggered
        /// the rejection.
        name: String,
    },
    /// A WIT type kind we don't support yet (futures, streams,
    /// error-context, owned/borrowed handles).
    #[error("unsupported WIT type kind: {kind}")]
    UnsupportedKind {
        /// Human-readable label for the unsupported kind
        /// (`"future"`, `"stream"`, `"map"`, etc.).
        kind: &'static str,
    },
}

/// Decode `bytes` and walk the root component's exports into a
/// [`LibrarySurface`].
///
/// Streams the bytes through a [`Validator`] so that we get fully
/// resolved [`ComponentEntityType`] values back from
/// [`TypesRef::component_entity_type_of_export`] — no separate WIT
/// round-trip is required.
pub fn extract_library_surface(bytes: &[u8]) -> Result<LibrarySurface, LibraryExtractError> {
    let walk = walk_component(bytes)?;
    let types = walk.types;
    let r = types.as_ref();
    let docs = parse_package_docs(walk.package_docs.as_deref());

    let mut items: Vec<LibraryItem> = Vec::new();
    for name in &walk.root_exports {
        let entity = r
            .component_entity_type_of_export(name)
            .ok_or_else(|| LibraryExtractError::Decode(format!("export `{name}` not in types")))?;
        match entity {
            ComponentEntityType::Func(fid) => {
                let func_doc = docs.get("").and_then(|m| m.get(name.as_str())).cloned();
                match build_func_decl(&r, name, func_type(&r, fid), func_doc) {
                    Ok(decl) => items.push(LibraryItem::Func(decl)),
                    Err(e @ LibraryExtractError::Resource { .. }) => return Err(e),
                    Err(_) => {}
                }
            }
            ComponentEntityType::Instance(iid) => {
                let short = iface_short_name(name);
                if short == "exports" {
                    continue;
                }
                let iface_ty = instance_type(&r, iid);
                let iface_docs = docs.get(short.as_str());
                match build_interface_funcs(&r, iface_ty, iface_docs) {
                    Ok(funcs) if !funcs.is_empty() => {
                        items.push(LibraryItem::Interface {
                            name: short,
                            export_name: name.clone(),
                            doc: None,
                            funcs,
                        });
                    }
                    Err(e @ LibraryExtractError::Resource { .. }) => return Err(e),
                    Ok(_) | Err(_) => {}
                }
            }
            // Module / Value / Type / Component exports aren't
            // dispatchable through a CLI.
            _ => {}
        }
    }

    Ok(LibrarySurface { items })
}

/// Output of the streaming walk over a component's bytes.
struct WalkedComponent {
    /// Validator output, the source of truth for all type queries.
    types: wasmparser::types::Types,
    /// Export names of the root component, in declaration order.
    root_exports: Vec<String>,
    /// Raw bytes of the `package-docs` custom section, if present.
    package_docs: Option<Vec<u8>>,
}

/// Stream `bytes` through a [`Validator`] and collect the root
/// component's export names plus its `package-docs` custom section.
///
/// The validator returns the fully-populated [`wasmparser::types::Types`]
/// from its [`ValidPayload::End`] return on the outermost end payload;
/// we capture that and never call `validator.end()` ourselves.
fn walk_component(bytes: &[u8]) -> Result<WalkedComponent, LibraryExtractError> {
    let mut validator = Validator::new_with_features(WasmFeatures::all());
    let mut depth: u32 = 0;
    let mut root_exports: Vec<String> = Vec::new();
    let mut package_docs: Option<Vec<u8>> = None;
    let mut types: Option<wasmparser::types::Types> = None;

    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.map_err(|e| LibraryExtractError::Decode(e.to_string()))?;
        match &payload {
            Payload::Version { encoding, .. } => {
                if depth == 0 && *encoding != Encoding::Component {
                    return Err(LibraryExtractError::NotAComponent);
                }
                depth += 1;
            }
            Payload::End(_) => {
                depth = depth.saturating_sub(1);
            }
            Payload::ComponentExportSection(reader) if depth == 1 => {
                for export in reader.clone() {
                    let export = export.map_err(|e| LibraryExtractError::Decode(e.to_string()))?;
                    root_exports.push(export.name.0.to_string());
                }
            }
            Payload::CustomSection(cs) if depth == 1 && cs.name() == "package-docs" => {
                package_docs = Some(cs.data().to_vec());
            }
            _ => {}
        }
        let valid = validator
            .payload(&payload)
            .map_err(|e| LibraryExtractError::Decode(e.to_string()))?;
        if let ValidPayload::End(t) = valid {
            types = Some(t);
        }
    }

    let types = types
        .ok_or_else(|| LibraryExtractError::Decode("validator produced no types".to_string()))?;
    Ok(WalkedComponent {
        types,
        root_exports,
        package_docs,
    })
}

/// Build a [`FuncDecl`] from a validated [`ComponentFuncType`].
fn build_func_decl(
    r: &TypesRef<'_>,
    name: &str,
    func_ty: &ComponentFuncType,
    doc: Option<String>,
) -> Result<FuncDecl, LibraryExtractError> {
    let mut params = Vec::with_capacity(func_ty.params.len());
    for (pname, pty) in &func_ty.params {
        params.push(ParamDecl {
            name: pname.to_string(),
            ty: cval_to_wit(r, pty)?,
        });
    }
    let results = match &func_ty.result {
        Some(ty) => vec![ResultDecl {
            ty: cval_to_wit(r, ty)?,
        }],
        None => Vec::new(),
    };
    Ok(FuncDecl {
        name: name.to_string(),
        doc,
        params,
        results,
    })
}

/// Build [`FuncDecl`]s for every function export of an instance.
///
/// Non-function exports (nested types, sub-instances) are skipped;
/// individual functions that use unsupported types are also skipped
/// so a single odd export doesn't poison the whole interface. Handle
/// (resource) types still surface as [`LibraryExtractError::Resource`]
/// because they signal a fundamentally non-CLI-compatible shape.
fn build_interface_funcs(
    r: &TypesRef<'_>,
    iface_ty: &ComponentInstanceType,
    func_docs: Option<&HashMap<String, String>>,
) -> Result<Vec<FuncDecl>, LibraryExtractError> {
    let mut funcs = Vec::new();
    for (fname, entity) in &iface_ty.exports {
        let ComponentEntityType::Func(fid) = entity else {
            continue;
        };
        let doc = func_docs.and_then(|m| m.get(fname.as_str())).cloned();
        match build_func_decl(r, fname, func_type(r, *fid), doc) {
            Ok(decl) => funcs.push(decl),
            Err(e @ LibraryExtractError::Resource { .. }) => return Err(e),
            Err(_) => {}
        }
    }
    Ok(funcs)
}

/// Map a [`ComponentValType`] to our local [`WitTy`] IR.
fn cval_to_wit(r: &TypesRef<'_>, vt: &ComponentValType) -> Result<WitTy, LibraryExtractError> {
    match vt {
        ComponentValType::Primitive(p) => prim_to_wit(*p),
        ComponentValType::Type(id) => defined_to_wit(r, defined_type(r, *id)),
    }
}

/// Look up a [`ComponentFuncType`] by id.
///
/// Type ids surfaced by `wasmparser::Validator` always resolve, so
/// the underlying `Index` impl can't panic on us — the indirection
/// here just isolates the one place clippy would otherwise flag.
#[allow(clippy::indexing_slicing)]
fn func_type<'a>(r: &'a TypesRef<'_>, id: ComponentFuncTypeId) -> &'a ComponentFuncType {
    &r[id]
}

/// Look up a [`ComponentInstanceType`] by id. See [`func_type`].
#[allow(clippy::indexing_slicing)]
fn instance_type<'a>(
    r: &'a TypesRef<'_>,
    id: ComponentInstanceTypeId,
) -> &'a ComponentInstanceType {
    &r[id]
}

/// Look up a [`ComponentDefinedType`] by id. See [`func_type`].
#[allow(clippy::indexing_slicing)]
fn defined_type<'a>(r: &'a TypesRef<'_>, id: ComponentDefinedTypeId) -> &'a ComponentDefinedType {
    &r[id]
}

/// Map a [`PrimitiveValType`] to our local [`WitTy`] IR.
fn prim_to_wit(p: PrimitiveValType) -> Result<WitTy, LibraryExtractError> {
    Ok(match p {
        PrimitiveValType::Bool => WitTy::Bool,
        PrimitiveValType::S8 => WitTy::S8,
        PrimitiveValType::S16 => WitTy::S16,
        PrimitiveValType::S32 => WitTy::S32,
        PrimitiveValType::S64 => WitTy::S64,
        PrimitiveValType::U8 => WitTy::U8,
        PrimitiveValType::U16 => WitTy::U16,
        PrimitiveValType::U32 => WitTy::U32,
        PrimitiveValType::U64 => WitTy::U64,
        PrimitiveValType::F32 => WitTy::F32,
        PrimitiveValType::F64 => WitTy::F64,
        PrimitiveValType::Char => WitTy::Char,
        PrimitiveValType::String => WitTy::String,
        PrimitiveValType::ErrorContext => {
            return Err(LibraryExtractError::UnsupportedKind {
                kind: "error-context",
            });
        }
    })
}

/// Map a [`ComponentDefinedType`] to our local [`WitTy`] IR.
fn defined_to_wit(
    r: &TypesRef<'_>,
    d: &ComponentDefinedType,
) -> Result<WitTy, LibraryExtractError> {
    match d {
        ComponentDefinedType::Primitive(p) => prim_to_wit(*p),
        ComponentDefinedType::Record(rec) => {
            let mut fields = Vec::with_capacity(rec.fields.len());
            for (fname, fty) in &rec.fields {
                fields.push((fname.to_string(), cval_to_wit(r, fty)?));
            }
            Ok(WitTy::Record(fields))
        }
        ComponentDefinedType::Variant(v) => {
            let mut cases = Vec::with_capacity(v.cases.len());
            for (cname, case) in &v.cases {
                let payload = match &case.ty {
                    Some(t) => Some(Box::new(cval_to_wit(r, t)?)),
                    None => None,
                };
                cases.push((cname.to_string(), payload));
            }
            Ok(WitTy::Variant(cases))
        }
        ComponentDefinedType::List(t) => Ok(WitTy::List(Box::new(cval_to_wit(r, t)?))),
        ComponentDefinedType::Option(t) => Ok(WitTy::Option(Box::new(cval_to_wit(r, t)?))),
        ComponentDefinedType::Tuple(t) => {
            let mut tys = Vec::with_capacity(t.types.len());
            for v in &t.types {
                tys.push(cval_to_wit(r, v)?);
            }
            Ok(WitTy::Tuple(tys))
        }
        ComponentDefinedType::Enum(s) => {
            Ok(WitTy::Enum(s.iter().map(ToString::to_string).collect()))
        }
        ComponentDefinedType::Flags(s) => {
            Ok(WitTy::Flags(s.iter().map(ToString::to_string).collect()))
        }
        ComponentDefinedType::Result { ok, err } => {
            let ok = match ok {
                Some(t) => Some(Box::new(cval_to_wit(r, t)?)),
                None => None,
            };
            let err = match err {
                Some(t) => Some(Box::new(cval_to_wit(r, t)?)),
                None => None,
            };
            Ok(WitTy::Result { ok, err })
        }
        ComponentDefinedType::Own(_) | ComponentDefinedType::Borrow(_) => {
            Err(LibraryExtractError::Resource {
                name: "<anonymous>".to_string(),
            })
        }
        ComponentDefinedType::Future(_) => {
            Err(LibraryExtractError::UnsupportedKind { kind: "future" })
        }
        ComponentDefinedType::Stream(_) => {
            Err(LibraryExtractError::UnsupportedKind { kind: "stream" })
        }
        ComponentDefinedType::Map(_, _) => {
            Err(LibraryExtractError::UnsupportedKind { kind: "map" })
        }
        ComponentDefinedType::FixedLengthList(_, _) => Err(LibraryExtractError::UnsupportedKind {
            kind: "fixed-length-list",
        }),
    }
}

/// Extract the short, user-facing interface name from a fully-
/// qualified export name like `local:time-server/time@0.1.0`.
///
/// Falls back to the full name when the format doesn't match (e.g.
/// inline `WorldKey::Name` style exports).
fn iface_short_name(export_name: &str) -> String {
    let after_slash = export_name.rsplit('/').next().unwrap_or(export_name);
    after_slash
        .split('@')
        .next()
        .unwrap_or(after_slash)
        .to_string()
}

/// Parse a `package-docs` custom section into a flat map of
/// `interface short name -> { func name -> doc string }`.
///
/// The map uses the empty string `""` as the key for top-level
/// (free) function docs. Unknown JSON shapes are silently ignored
/// — docs are a best-effort enhancement, never a hard requirement.
fn parse_package_docs(raw: Option<&[u8]>) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    let Some(bytes) = raw else { return out };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return out;
    };
    if let Some(top_funcs) = json.get("funcs").and_then(|v| v.as_object()) {
        let entry = out.entry(String::new()).or_default();
        for (fname, fobj) in top_funcs {
            if let Some(doc) = fobj.get("docs").and_then(|d| d.as_str()) {
                entry.insert(fname.clone(), doc.to_string());
            }
        }
    }
    if let Some(ifaces) = json.get("interfaces").and_then(|v| v.as_object()) {
        for (iname, iobj) in ifaces {
            let Some(funcs) = iobj.get("funcs").and_then(|v| v.as_object()) else {
                continue;
            };
            let entry = out.entry(iname.clone()).or_default();
            for (fname, fobj) in funcs {
                if let Some(doc) = fobj.get("docs").and_then(|d| d.as_str()) {
                    entry.insert(fname.clone(), doc.to_string());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    fn read_fixture(name: &str) -> Vec<u8> {
        std::fs::read(fixture_path(name)).expect("read fixture")
    }

    // r[verify run.library-detection]
    #[test]
    fn extract_wordmark_surface() {
        let bytes = read_fixture("library_wordmark.wasm");
        let surface = extract_library_surface(&bytes).expect("extract");
        assert_eq!(surface.items.len(), 1);
        let LibraryItem::Func(decl) = &surface.items[0] else {
            panic!("expected free function, got {:?}", surface.items[0]);
        };
        assert_eq!(decl.name, "to-word");
        assert_eq!(decl.params.len(), 1);
        assert_eq!(decl.params[0].name, "markdown");
        assert!(matches!(decl.params[0].ty, WitTy::String));
        assert_eq!(decl.results.len(), 1);
        assert!(matches!(
            decl.results[0].ty,
            WitTy::Result {
                ok: Some(_),
                err: Some(_)
            }
        ));
    }

    // r[verify run.library-dispatch]
    #[test]
    fn extract_kitchen_sink_surface() {
        let bytes = read_fixture("library_kitchen_sink.wasm");
        let surface = extract_library_surface(&bytes).expect("extract");

        // Must contain at least one interface (math) plus the free
        // functions.
        let has_iface = surface
            .items
            .iter()
            .any(|i| matches!(i, LibraryItem::Interface { .. }));
        assert!(has_iface, "expected math interface in surface");

        let names: Vec<&str> = surface
            .items
            .iter()
            .map(|i| match i {
                LibraryItem::Func(f) => f.name.as_str(),
                LibraryItem::Interface { name, .. } => name.as_str(),
            })
            .collect();
        for expected in &["shout", "greet", "pick", "fail"] {
            assert!(
                names.iter().any(|n| *n == *expected),
                "missing export {expected}; got {names:?}"
            );
        }
    }

    // r[verify run.library-resources-rejected]
    #[test]
    fn extract_resources_fixture_is_rejected() {
        let bytes = read_fixture("library_resources.wasm");
        let err = extract_library_surface(&bytes).expect_err("must reject resource");
        assert!(
            matches!(err, LibraryExtractError::Resource { .. }),
            "expected Resource error, got {err:?}"
        );
    }
}
