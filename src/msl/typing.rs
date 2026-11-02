// The types of a straight-line body's values (see the top of mod.rs): each definition is a
// value (moves, loads and saves of slots pass values on), so the body is in SSA form, and the
// backward slices of the denominators are one reverse sweep over the values.

use std::collections::HashMap;

use super::{Item, MslCounts, Op, Placement};

/// A value's type: masks (comparison results, sign bits) are 32-bit in every type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    F32,
    D64,
    Mask,
}

pub(super) struct Types {
    /// per item: the type of the value it defines (or copies), and those of its operands
    pub items: Vec<(Ty, Vec<Ty>)>,
    /// the value types of the memory slots at the end (the outputs scattered there)
    pub final_mem: Vec<Ty>,
    pub counts: MslCounts,
}

struct Value {
    mask: bool,
    args: Vec<usize>,
    /// df64 whatever its uses: a df64 input, a division or a reciprocal
    forced: bool,
    /// the operands that must be df64: a divisor, a reciprocal's operand
    seeds: Vec<usize>,
}

struct Values {
    values: Vec<Value>,
    vars: HashMap<String, usize>,
}

impl Values {
    fn add(&mut self, mask: bool, args: Vec<usize>, forced: bool, seeds: Vec<usize>) -> usize {
        self.values.push(Value {
            mask,
            args,
            forced,
            seeds,
        });
        self.values.len() - 1
    }

    /// The value of `var`: a zero (the variables start at 0) before its first definition.
    fn get(&mut self, var: &str) -> usize {
        if let Some(&v) = self.vars.get(var) {
            return v;
        }
        let v = self.add(false, Vec::new(), false, Vec::new());
        self.vars.insert(var.to_string(), v);
        v
    }
}

/// The types of `body` (after `inline_funclets`); `f64_inputs` is None without mixed
/// precision, when every value is f32.
pub(super) fn types(
    body: &[Item],
    place: &Placement,
    count_states: usize,
    count_mem: usize,
    f64_inputs: Option<&[bool]>,
    row_first: bool,
) -> Types {
    let mixed = f64_inputs.is_some();
    let f64_input = |k: usize| f64_inputs.is_some_and(|f| f.get(k).copied().unwrap_or(false));
    let mut v = Values {
        values: Vec::new(),
        vars: HashMap::new(),
    };
    // the memory slots that are locals: gathered inputs, or zeros
    for k in 0..count_mem {
        if !place.direct_in[k] && !place.direct_out[k] {
            let x = v.add(
                false,
                Vec::new(),
                k < count_states && f64_input(k),
                Vec::new(),
            );
            v.vars.insert(format!("m{}", k), x);
        }
    }

    // forward: the values each item defines and uses
    let mut items: Vec<(Option<usize>, Vec<usize>)> = Vec::with_capacity(body.len());
    for item in body {
        let entry = match item {
            Item::Op(dst, op, args) => {
                let a: Vec<usize> = args.iter().map(|x| v.get(x)).collect();
                let def = if *op == Op::Mov {
                    a[0]
                } else {
                    let value = |k: usize, v: &Values| !v.values[a[k]].mask;
                    let mask = match op {
                        Op::Cmp(_) | Op::Sign => true,
                        Op::And | Op::Or | Op::Xor | Op::AndNot | Op::Not => {
                            !(0..a.len()).any(|k| value(k, &v))
                        }
                        Op::Select => !(value(0, &v) || value(1, &v)),
                        _ => false,
                    };
                    let (forced, seeds) = match op {
                        Op::Div if mixed => (true, vec![a[1]]),
                        Op::Recip if mixed => (true, vec![a[0]]),
                        _ => (false, Vec::new()),
                    };
                    v.add(mask, a.clone(), forced, seeds)
                };
                v.vars.insert(dst.clone(), def);
                (Some(def), a)
            }
            Item::Const(dst, _) => {
                let x = v.add(false, Vec::new(), false, Vec::new());
                v.vars.insert(dst.clone(), x);
                (Some(x), Vec::new())
            }
            Item::LoadParam(dst, idx) => {
                let forced = row_first && f64_input(*idx as usize);
                let x = v.add(false, Vec::new(), forced, Vec::new());
                v.vars.insert(dst.clone(), x);
                (Some(x), Vec::new())
            }
            Item::LoadMem(dst, idx) => {
                let k = *idx as usize;
                let x = if place.direct_in[k] {
                    v.add(false, Vec::new(), f64_input(k), Vec::new())
                } else {
                    v.get(&format!("m{}", k))
                };
                v.vars.insert(dst.clone(), x);
                (Some(x), Vec::new())
            }
            Item::SaveMem(src, idx) => {
                let k = *idx as usize;
                let x = v.get(src);
                if place.direct_out[k] {
                    (None, vec![x])
                } else {
                    v.vars.insert(format!("m{}", k), x);
                    (Some(x), vec![x])
                }
            }
            Item::LoadStack(dst, idx) => {
                let x = v.get(&format!("s{}", idx));
                v.vars.insert(dst.clone(), x);
                (Some(x), Vec::new())
            }
            Item::SaveStack(src, idx) => {
                let x = v.get(src);
                v.vars.insert(format!("s{}", idx), x);
                (Some(x), vec![x])
            }
            Item::Label(_) | Item::Branch(_) | Item::CallFunclet(_) | Item::Ret => {
                (None, Vec::new())
            }
        };
        items.push(entry);
    }

    // backward: a value's operands come before it, so one reverse sweep marks the slices
    let n = v.values.len();
    let mut marked = vec![false; n];
    for x in (0..n).rev() {
        for &s in &v.values[x].seeds {
            marked[s] = true;
        }
        if marked[x] && !v.values[x].mask {
            for k in 0..v.values[x].args.len() {
                let a = v.values[x].args[k];
                if !v.values[a].mask {
                    marked[a] = true;
                }
            }
        }
    }
    let ty = |x: usize| {
        let value = &v.values[x];
        if value.mask {
            Ty::Mask
        } else if mixed && (marked[x] || value.forced) {
            Ty::D64
        } else {
            Ty::F32
        }
    };

    let mut counts = MslCounts::default();
    for (item, (def, _)) in body.iter().zip(&items) {
        if let (Item::Op(_, op, _), Some(d)) = (item, def) {
            if *op != Op::Mov {
                if ty(*d) == Ty::D64 {
                    counts.df64_ops += 1;
                } else {
                    counts.f32_ops += 1;
                }
            }
        }
    }
    let final_mem = (0..count_mem)
        .map(|k| v.vars.get(&format!("m{}", k)).map_or(Ty::F32, |&x| ty(x)))
        .collect();
    Types {
        items: items
            .iter()
            .map(|(def, args)| {
                (
                    def.map_or(Ty::F32, ty),
                    args.iter().map(|&a| ty(a)).collect(),
                )
            })
            .collect(),
        final_mem,
        counts,
    }
}
