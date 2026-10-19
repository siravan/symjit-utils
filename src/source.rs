//! Rust source input (`rust2sjb`): compiles a `#[kernel]` function of a Rust file (the language
//! of the development line's `composer` feature and its `#[kernel]` macro) without a Rust
//! compiler. The file is parsed with `syn` and the kernel's body is interpreted (`interp.rs`)
//! against `crate::kernel::KernelBuilder`, doing what the code the macro generates does, so that
//! the kernel is the one the macro builds. (From sjc, whose `source.rs` this is.)
//!
//! Supported: everything the macro accepts in kernels, `#[kernel(inline)]` helpers and
//! recursion, except `match`, macros, and Rust beyond numbers, tuples, arrays, ranges and simple
//! closures. Calls of other `#[kernel]` kernels and of `#[kernel(callback)]` functions are
//! rejected: a saved application holds one kernel and no Rust functions.

mod interp;
mod syntax;
mod value;

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use symjit::{Application, CompilerType, Config};

use crate::kernel::{Cplx, KernelBuilder, Real, Value};

use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Expr, FnArg, Ident, ItemFn, Pat, ReturnType, Token, UseTree};

use interp::{at, float_const, Flow, Interp};
use syntax::{stage_type, HasWhile, Ty};
use value::{lift, to_slots, Slots, Val};

/// The configuration kernels are compiled with by default: that of `Config::default()` without
/// a `symjit.toml` (and of the development line's `Config::builtin`): opt level 2, fastmath,
/// common-subexpression elimination, fast complex code, for `ty`.
pub fn builtin_config(ty: CompilerType) -> Result<Config> {
    let mut config = Config::new(ty, 0)?;
    config.set_simd(true);
    config.set_cse(true);
    config.set_symbolica(true);
    config.set_compact(true);
    config.set_fastmath(true);
    config.set_fast_complex(true);
    config.set_parallel_mul(true);
    config.set_opt_level(2);
    config.set_direct_arena_identity_output(true);
    Ok(config)
}

/// What the `#[kernel]` attribute makes of a function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FnKind {
    /// `#[kernel]`: compiled as a kernel
    Kernel,
    /// `#[kernel(inline)]`: inlined into the kernels that call it
    Inline,
    /// `#[kernel(callback)]`: Rust called by kernels at run time
    Callback,
    /// no attribute
    Plain,
}

pub struct FnDef {
    pub name: String,
    pub kind: FnKind,
    /// `#[kernel(complex)]`
    pub complex: bool,
    pub item: ItemFn,
    pub params: Vec<(Pat, Ty)>,
    pub ret: Option<Ty>,
}

/// A parsed Rust file.
pub struct Source {
    pub path: String,
    pub fns: HashMap<String, FnDef>,
    /// the functions in the order of the file
    order: Vec<String>,
    /// the constants of the file (`const N: usize = 4;`)
    pub consts: HashMap<String, Expr>,
    /// the names imported from `std::f64::consts` (`use std::f64::consts::PI;`)
    pub imported: HashMap<String, f64>,
}

impl Source {
    pub fn parse(path: &str, text: &str) -> Result<Source> {
        let file = syn::parse_file(text).map_err(|e| {
            let p = e.span().start();
            anyhow!("{}:{}:{}: {}", path, p.line, p.column + 1, e)
        })?;
        let mut src = Source {
            path: path.to_string(),
            fns: HashMap::new(),
            order: Vec::new(),
            consts: HashMap::new(),
            imported: HashMap::new(),
        };

        for item in file.items.iter() {
            match item {
                syn::Item::Const(c) => {
                    src.consts.insert(c.ident.to_string(), (*c.expr).clone());
                }
                syn::Item::Use(u) => src.imports(&u.tree, &mut Vec::new()),
                _ => {}
            }
        }
        for item in file.items.into_iter() {
            if let syn::Item::Fn(f) = item {
                let def = src.fn_def(f)?;
                src.order.push(def.name.clone());
                src.fns.insert(def.name.clone(), def);
            }
        }
        Ok(src)
    }

    /// Records the names a `use` imports from `std::f64::consts`.
    fn imports(&mut self, tree: &UseTree, prefix: &mut Vec<String>) {
        let consts = |p: &[String]| p.last().map(|s| s.as_str()) == Some("consts");
        match tree {
            UseTree::Path(p) => {
                prefix.push(p.ident.to_string());
                self.imports(&p.tree, prefix);
                prefix.pop();
            }
            UseTree::Name(n) if consts(prefix) => {
                let name = n.ident.to_string();
                if let Some(x) = float_const(&name) {
                    self.imported.insert(name, x);
                }
            }
            UseTree::Rename(r) if consts(prefix) => {
                if let Some(x) = float_const(&r.ident.to_string()) {
                    self.imported.insert(r.rename.to_string(), x);
                }
            }
            UseTree::Glob(_) if consts(prefix) => {
                for name in [
                    "PI",
                    "TAU",
                    "E",
                    "SQRT_2",
                    "FRAC_1_SQRT_2",
                    "LN_2",
                    "LN_10",
                    "LOG2_E",
                    "LOG10_E",
                    "LOG2_10",
                    "LOG10_2",
                    "FRAC_PI_2",
                    "FRAC_PI_3",
                    "FRAC_PI_4",
                    "FRAC_PI_6",
                    "FRAC_PI_8",
                    "FRAC_1_PI",
                    "FRAC_2_PI",
                    "FRAC_2_SQRT_PI",
                ] {
                    self.imported
                        .insert(name.to_string(), float_const(name).unwrap());
                }
            }
            UseTree::Group(g) => {
                for t in g.items.iter() {
                    self.imports(t, prefix);
                }
            }
            _ => {}
        }
    }

    fn err(&self, span: proc_macro2::Span, msg: impl std::fmt::Display) -> anyhow::Error {
        anyhow!("{}:{}", self.path, at(span, msg))
    }

    fn fn_def(&self, item: ItemFn) -> Result<FnDef> {
        let name = item.sig.ident.to_string();
        let mut kind = FnKind::Plain;
        let mut complex = false;
        for attr in item.attrs.iter() {
            let last = attr.path().segments.last().map(|s| s.ident.to_string());
            if last.as_deref() != Some("kernel") {
                continue;
            }
            kind = FnKind::Kernel;
            if let syn::Meta::List(_) = &attr.meta {
                let idents = attr
                    .parse_args_with(Punctuated::<Ident, Token![,]>::parse_terminated)
                    .map_err(|e| self.err(e.span(), e))?;
                for id in idents {
                    match id.to_string().as_str() {
                        "inline" => kind = FnKind::Inline,
                        "callback" => kind = FnKind::Callback,
                        "complex" => complex = true,
                        _ => {
                            return Err(
                                self.err(id.span(), "expected `inline`, `complex` or `callback`")
                            )
                        }
                    }
                }
            }
        }

        let mut params = Vec::new();
        for arg in item.sig.inputs.iter() {
            match arg {
                FnArg::Typed(pt) => params.push((
                    (*pt.pat).clone(),
                    stage_type(&pt.ty, &|e| self.const_usize(e)),
                )),
                FnArg::Receiver(r) if kind != FnKind::Plain => {
                    return Err(self.err(r.span(), "a kernel cannot be a method"))
                }
                FnArg::Receiver(_) => {}
            }
        }
        let ret = match &item.sig.output {
            ReturnType::Type(_, ty) => Some(stage_type(ty, &|e| self.const_usize(e))),
            ReturnType::Default => None,
        };
        Ok(FnDef {
            name,
            kind,
            complex,
            item,
            params,
            ret,
        })
    }

    /// The value of an array length: an integer literal or a constant of the file defined by
    /// one.
    pub fn const_usize(&self, e: &Expr) -> Option<usize> {
        match e {
            Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(i),
                ..
            }) => i.base10_parse().ok(),
            Expr::Path(p) => {
                let id = p.path.get_ident()?.to_string();
                self.consts.get(&id).and_then(|e| self.const_usize(e))
            }
            Expr::Paren(p) => self.const_usize(&p.expr),
            Expr::Group(g) => self.const_usize(&g.expr),
            _ => None,
        }
    }

    /// The `#[kernel]` kernels of the file (not helpers or callbacks), in order.
    pub fn kernels(&self) -> Vec<&str> {
        self.order
            .iter()
            .filter(|n| self.fns[*n].kind == FnKind::Kernel)
            .map(|n| n.as_str())
            .collect()
    }

    /// Compiles the kernel `name` with `config` (made complex if the kernel is, and set to the
    /// direct translator if it has a `while`), as `name::compile(config)` of the macro does.
    pub fn compile(&self, name: &str, mut config: Config) -> Result<Application> {
        let Some(def) = self.fns.get(name).filter(|d| d.kind == FnKind::Kernel) else {
            let kernels = self.kernels();
            return Err(anyhow!(
                "{}: no `#[kernel]` named `{}`{}",
                self.path,
                name,
                if kernels.is_empty() {
                    String::new()
                } else {
                    format!(" (the kernels are: {})", kernels.join(", "))
                }
            ));
        };
        self.check_signature(def)?;
        self.build_kernel(def, &mut config)
    }

    fn build_kernel(&self, def: &FnDef, config: &mut Config) -> Result<Application> {
        let name = def.name.as_str();
        let ret = def.ret.clone().unwrap();
        let complex =
            def.complex || ret.has_complex() || def.params.iter().any(|(_, t)| t.has_complex());
        let mut has_while = HasWhile::default();
        has_while.visit_block(&def.item.block);
        config.set_complex(complex);
        if has_while.0 {
            // kernel loops need the direct translator
            config.set_direct(true);
        }

        let k = KernelBuilder::new(config.clone());
        let interp = Interp::new(&k, self, def);
        let result = (|| -> Result<Vec<symjit::Slot>, Flow> {
            for (pat, ty) in def.params.iter() {
                let v = params(&k, ty);
                interp.bind(pat, v)?;
            }
            let v = interp.eval_block(&def.item.block)?;
            let span = def.item.sig.output.span();
            let out = lift(&k, v, &ret)
                .map_err(|msg| Flow::Error(at(span, format!("the kernel's value: {}", msg))))?;
            let mut slots = Vec::new();
            to_slots(&k, &out, false, &mut slots).map_err(|msg| Flow::Error(at(span, msg)))?;
            Ok(slots)
        })();
        let slots = match result {
            Ok(slots) => slots,
            Err(Flow::Error(msg)) => return Err(anyhow!("{}:{}", self.path, msg)),
            Err(_) => {
                return Err(anyhow!(
                    "{}: `break` or `continue` outside a loop in `{}`",
                    self.path,
                    name
                ))
            }
        };
        k.compile(Slots::new(slots))
            .map_err(|e| anyhow!("{}: the kernel `{}`: {}", self.path, name, e))
    }

    /// The checks of the macro on a kernel's signature.
    fn check_signature(&self, def: &FnDef) -> Result<()> {
        let sig = &def.item.sig;
        if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
            return Err(self.err(sig.generics.span(), "a kernel cannot be generic"));
        }
        if sig.asyncness.is_some() {
            return Err(self.err(sig.span(), "a kernel cannot be async"));
        }
        if sig.variadic.is_some() {
            return Err(self.err(sig.span(), "a kernel cannot be variadic"));
        }
        for ((_, ty), arg) in def.params.iter().zip(sig.inputs.iter()) {
            if !ty.has_handle() || ty.has_other() || ty.has_mask() {
                return Err(self.err(
                    arg.span(),
                    "the parameters of a kernel must be `f64` or `Complex<f64>` (or tuples or \
                     arrays of them); use `#[kernel(inline)]` for a helper with other parameters",
                ));
            }
        }
        match &def.ret {
            Some(ty) if ty.has_handle() && !ty.has_other() => Ok(()),
            _ => Err(self.err(
                sig.ident.span(),
                "a kernel must return `f64`, `Complex<f64>` or `bool` (or tuples or arrays of them)",
            )),
        }
    }
}

/// The parameters of type `ty`, declared in order (`Value::from_params`).
fn params<'k>(k: &'k KernelBuilder, ty: &Ty) -> Val<'k> {
    match ty {
        Ty::Real => Val::Real(Real::from_params(k)),
        Ty::Cplx => Val::Cplx(Cplx::from_params(k)),
        Ty::Tuple(ts) => Val::Tuple(ts.iter().map(|t| params(k, t)).collect()),
        Ty::Array(t, n) => Val::Array((0..*n).map(|_| params(k, t)).collect()),
        _ => Val::Unit,
    }
}
