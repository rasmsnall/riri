//! A MIR interpreter, so far for one thread and scalars only.
//!
//! This is the second step of the MIR work. The driver could already reach
//! monomorphized MIR; this evaluates it. What it covers is the shape a kernel
//! body actually has once rustc is done with it: locals, assignments, checked
//! arithmetic with its overflow `Assert`, comparisons, `SwitchInt`, and casts.
//!
//! What it does not cover yet is everything that makes it a *GPU* interpreter:
//! memory with provenance, calls, and the SIMT layer that would run one of
//! these per lane against the scheduler the library already has. Those are the
//! next steps and they are much larger than this one.

use rustc_public::mir::{
    BinOp, Body, ConstOperand, Operand, Place, ProjectionElem, Rvalue, RuntimeChecks,
    StatementKind, TerminatorKind, UnOp,
};
use rustc_public::ty::{ConstantKind, IntTy, RigidTy, Ty, TyKind, UintTy};

use crate::value::{IntKind, Value};

/// Guards against a kernel that never terminates, which an interpreter cannot
/// tell from one that is merely slow.
const STEP_LIMIT: usize = 1_000_000;

pub struct Interp {
    locals: Vec<Value>,
    steps: usize,
}

impl Interp {
    /// Runs a body with no arguments and returns local 0, the return place.
    pub fn run(body: &Body) -> Result<Value, String> {
        let mut interp = Interp {
            locals: vec![Value::Uninit; body.locals().len()],
            steps: 0,
        };
        interp.execute(body)
    }

    fn execute(&mut self, body: &Body) -> Result<Value, String> {
        let mut block = 0usize;
        loop {
            self.steps += 1;
            if self.steps > STEP_LIMIT {
                return Err(format!("gave up after {STEP_LIMIT} steps"));
            }

            let Some(bb) = body.blocks.get(block) else {
                return Err(format!("jumped to bb{block}, which does not exist"));
            };

            for statement in &bb.statements {
                self.statement(&statement.kind)?;
            }

            match &bb.terminator.kind {
                TerminatorKind::Goto { target } => block = *target,
                TerminatorKind::Return => return Ok(self.locals[0].clone()),
                TerminatorKind::SwitchInt { discr, targets } => {
                    let value = self.operand(discr)?.discriminant()?;
                    block = targets
                        .branches()
                        .find(|(arm, _)| *arm == value)
                        .map(|(_, target)| target)
                        .unwrap_or_else(|| targets.otherwise());
                }
                TerminatorKind::Assert {
                    cond,
                    expected,
                    target,
                    msg,
                    ..
                } => {
                    let held = self.operand(cond)?.as_bool()?;
                    if held != *expected {
                        // This is where an overflow or a bounds check lands.
                        // rustc has already worked out which, and says so in
                        // the words the panic would have used.
                        let why = msg
                            .description()
                            .map(str::to_string)
                            .unwrap_or_else(|_| format!("{msg:?}"));
                        return Err(why);
                    }
                    block = *target;
                }
                TerminatorKind::Call { func, .. } => {
                    return Err(format!("calls are not interpreted yet: {func:?}"));
                }
                TerminatorKind::Unreachable => {
                    return Err("reached unreachable".to_string());
                }
                other => return Err(format!("terminator not interpreted yet: {other:?}")),
            }
        }
    }

    fn statement(&mut self, kind: &StatementKind) -> Result<(), String> {
        match kind {
            StatementKind::Assign(place, rvalue) => {
                let value = self.rvalue(rvalue)?;
                self.store(place, value)
            }
            // Liveness and diagnostics carry no runtime meaning here. Storage
            // markers would matter once memory exists; today a local is just
            // a slot that is either set or not.
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::FakeRead(..)
            | StatementKind::PlaceMention(_)
            | StatementKind::AscribeUserType { .. }
            | StatementKind::Coverage(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop => Ok(()),
            other => Err(format!("statement not interpreted yet: {other:?}")),
        }
    }

    fn rvalue(&mut self, rvalue: &Rvalue) -> Result<Value, String> {
        match rvalue {
            Rvalue::Use(operand, _) => self.operand(operand),
            Rvalue::BinaryOp(op, lhs, rhs) => {
                let lhs = self.operand(lhs)?;
                let rhs = self.operand(rhs)?;
                binary(*op, &lhs, &rhs).map(|(value, _)| value)
            }
            Rvalue::CheckedBinaryOp(op, lhs, rhs) => {
                let lhs = self.operand(lhs)?;
                let rhs = self.operand(rhs)?;
                let (value, overflowed) = binary(*op, &lhs, &rhs)?;
                Ok(Value::Aggregate(vec![value, Value::Bool(overflowed)]))
            }
            Rvalue::UnaryOp(op, operand) => {
                let value = self.operand(operand)?;
                unary(*op, &value)
            }
            Rvalue::Cast(_, operand, ty) => {
                let value = self.operand(operand)?;
                cast(&value, *ty)
            }
            Rvalue::Aggregate(_, operands) => {
                let parts: Result<Vec<_>, _> =
                    operands.iter().map(|o| self.operand(o)).collect();
                Ok(Value::Aggregate(parts?))
            }
            Rvalue::CopyForDeref(place) => self.load(place),
            other => Err(format!("rvalue not interpreted yet: {other:?}")),
        }
    }

    fn operand(&mut self, operand: &Operand) -> Result<Value, String> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.load(place),
            Operand::Constant(konst) => constant(konst),
            // Whether the build has overflow checks on. It does, because
            // that is what produced the `CheckedBinaryOp` above.
            Operand::RuntimeChecks(RuntimeChecks::OverflowChecks) => Ok(Value::Bool(true)),
            Operand::RuntimeChecks(_) => Ok(Value::Bool(false)),
        }
    }

    fn load(&self, place: &Place) -> Result<Value, String> {
        let mut value = self
            .locals
            .get(place.local)
            .ok_or_else(|| format!("no local _{}", place.local))?
            .clone();

        for element in &place.projection {
            value = match element {
                ProjectionElem::Field(index, _) => match value {
                    Value::Aggregate(parts) => parts
                        .get(*index)
                        .cloned()
                        .ok_or_else(|| format!("no field {index}"))?,
                    other => return Err(format!("cannot take a field of {other:?}")),
                },
                other => return Err(format!("projection not interpreted yet: {other:?}")),
            };
        }

        if value == Value::Uninit {
            return Err(format!("read of uninitialised _{}", place.local));
        }
        Ok(value)
    }

    fn store(&mut self, place: &Place, value: Value) -> Result<(), String> {
        let slot = self
            .locals
            .get_mut(place.local)
            .ok_or_else(|| format!("no local _{}", place.local))?;

        if place.projection.is_empty() {
            *slot = value;
            return Ok(());
        }

        // Only one level of field projection is supported, which is all the
        // bodies reached so far need.
        match place.projection.as_slice() {
            [ProjectionElem::Field(index, _)] => match slot {
                Value::Aggregate(parts) => {
                    let part = parts
                        .get_mut(*index)
                        .ok_or_else(|| format!("no field {index}"))?;
                    *part = value;
                    Ok(())
                }
                other => Err(format!("cannot assign a field of {other:?}")),
            },
            other => Err(format!("projection not interpreted yet: {other:?}")),
        }
    }
}

fn int_kind(ty: Ty) -> Option<IntKind> {
    // Pointer-sized types are modelled as 64 bit, matching the hosts this
    // runs on. A cross-target interpreter would take this from the target
    // spec instead.
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Uint(uint)) => Some(match uint {
            UintTy::Usize => IntKind::new(64, false),
            UintTy::U8 => IntKind::new(8, false),
            UintTy::U16 => IntKind::new(16, false),
            UintTy::U32 => IntKind::new(32, false),
            UintTy::U64 => IntKind::new(64, false),
            UintTy::U128 => IntKind::new(128, false),
        }),
        TyKind::RigidTy(RigidTy::Int(int)) => Some(match int {
            IntTy::Isize => IntKind::new(64, true),
            IntTy::I8 => IntKind::new(8, true),
            IntTy::I16 => IntKind::new(16, true),
            IntTy::I32 => IntKind::new(32, true),
            IntTy::I64 => IntKind::new(64, true),
            IntTy::I128 => IntKind::new(128, true),
        }),
        _ => None,
    }
}

fn constant(konst: &ConstOperand) -> Result<Value, String> {
    let ty = konst.const_.ty();
    match konst.const_.kind() {
        ConstantKind::Allocated(alloc) => {
            if matches!(ty.kind(), TyKind::RigidTy(RigidTy::Bool)) {
                let raw = alloc.read_uint().map_err(|e| format!("{e:?}"))?;
                return Ok(Value::Bool(raw != 0));
            }
            let kind =
                int_kind(ty).ok_or_else(|| format!("constant type not supported: {:?}", ty.kind()))?;
            let raw = alloc.read_uint().map_err(|e| format!("{e:?}"))?;
            Ok(Value::int(raw, kind))
        }
        other => Err(format!("constant not interpreted yet: {other:?}")),
    }
}

fn cast(value: &Value, ty: Ty) -> Result<Value, String> {
    let kind = int_kind(ty).ok_or_else(|| format!("cast target not supported: {:?}", ty.kind()))?;
    match value {
        Value::Bool(b) => Ok(Value::int(*b as u128, kind)),
        Value::Int { raw, kind: from } => {
            // Widening a signed value sign-extends; everything else is a
            // truncation, which the constructor does.
            let extended = if from.signed {
                from.as_signed(*raw) as u128
            } else {
                *raw
            };
            Ok(Value::int(extended, kind))
        }
        other => Err(format!("cannot cast {other:?}")),
    }
}

fn unary(op: UnOp, value: &Value) -> Result<Value, String> {
    match op {
        UnOp::Not => match value {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            Value::Int { raw, kind } => Ok(Value::int(!raw, *kind)),
            other => Err(format!("cannot negate {other:?}")),
        },
        UnOp::Neg => {
            let (raw, kind) = value.as_int()?;
            Ok(Value::int((raw as i128).wrapping_neg() as u128, kind))
        }
        UnOp::PtrMetadata => Err("pointer metadata needs a memory model".to_string()),
    }
}

/// Applies a binary operator, returning the value and whether it overflowed.
///
/// Overflow is only meaningful for the arithmetic cases; comparisons report
/// false, which is what `CheckedBinaryOp` would want if it ever saw one.
fn binary(op: BinOp, lhs: &Value, rhs: &Value) -> Result<(Value, bool), String> {
    if let (Value::Bool(a), Value::Bool(b)) = (lhs, rhs) {
        let value = match op {
            BinOp::Eq => a == b,
            BinOp::Ne => a != b,
            BinOp::BitAnd => *a && *b,
            BinOp::BitOr => *a || *b,
            BinOp::BitXor => a != b,
            other => return Err(format!("{other:?} is not defined on bools")),
        };
        return Ok((Value::Bool(value), false));
    }

    let (left, kind) = lhs.as_int()?;
    let (right, right_kind) = rhs.as_int()?;

    // Shifts take their amount from a possibly different type; everything
    // else in MIR is same-typed by construction.
    let shift = matches!(
        op,
        BinOp::Shl | BinOp::ShlUnchecked | BinOp::Shr | BinOp::ShrUnchecked
    );
    if !shift && kind != right_kind {
        return Err(format!("mismatched operand types: {kind:?} and {right_kind:?}"));
    }

    let a = kind.as_signed(left);
    let b = kind.as_signed(right);

    let compare = |ordering: std::cmp::Ordering| -> bool {
        if kind.signed {
            a.cmp(&b) == ordering
        } else {
            left.cmp(&right) == ordering
        }
    };
    use std::cmp::Ordering::{Greater, Less};

    let (value, overflowed) = match op {
        BinOp::Add | BinOp::AddUnchecked => checked(kind, a.wrapping_add(b), left, right, |x, y| {
            x.wrapping_add(y)
        }),
        BinOp::Sub | BinOp::SubUnchecked => checked(kind, a.wrapping_sub(b), left, right, |x, y| {
            x.wrapping_sub(y)
        }),
        BinOp::Mul | BinOp::MulUnchecked => checked(kind, a.wrapping_mul(b), left, right, |x, y| {
            x.wrapping_mul(y)
        }),
        BinOp::Div => {
            if right == 0 {
                return Err("division by zero".to_string());
            }
            let raw = if kind.signed {
                a.wrapping_div(b) as u128
            } else {
                left / right
            };
            (Value::int(raw, kind), false)
        }
        BinOp::Rem => {
            if right == 0 {
                return Err("remainder by zero".to_string());
            }
            let raw = if kind.signed {
                a.wrapping_rem(b) as u128
            } else {
                left % right
            };
            (Value::int(raw, kind), false)
        }
        BinOp::BitAnd => (Value::int(left & right, kind), false),
        BinOp::BitOr => (Value::int(left | right, kind), false),
        BinOp::BitXor => (Value::int(left ^ right, kind), false),
        BinOp::Shl | BinOp::ShlUnchecked => {
            let amount = right % kind.bits as u128;
            (Value::int(left << amount, kind), right >= kind.bits as u128)
        }
        BinOp::Shr | BinOp::ShrUnchecked => {
            let amount = right % kind.bits as u128;
            let raw = if kind.signed {
                (a >> amount) as u128
            } else {
                left >> amount
            };
            (Value::int(raw, kind), right >= kind.bits as u128)
        }
        BinOp::Eq => (Value::Bool(left == right), false),
        BinOp::Ne => (Value::Bool(left != right), false),
        BinOp::Lt => (Value::Bool(compare(Less)), false),
        BinOp::Gt => (Value::Bool(compare(Greater)), false),
        BinOp::Le => (Value::Bool(!compare(Greater)), false),
        BinOp::Ge => (Value::Bool(!compare(Less)), false),
        other => return Err(format!("operator not interpreted yet: {other:?}")),
    };
    Ok((value, overflowed))
}

/// Arithmetic plus the overflow flag `CheckedBinaryOp` wants.
///
/// Signed overflow is decided on the widened result; unsigned on the raw
/// wrapping, since a `u128` subtraction that went below zero cannot be seen
/// in the signed domain.
fn checked(
    kind: IntKind,
    signed_result: i128,
    left: u128,
    right: u128,
    wrapping: impl Fn(u128, u128) -> u128,
) -> (Value, bool) {
    let raw = wrapping(left, right);
    let overflowed = if kind.signed {
        !kind.fits(signed_result)
    } else {
        kind.truncate(raw) != raw || !fits_unsigned(kind, left, right, raw, &wrapping)
    };
    (Value::int(raw, kind), overflowed)
}

/// Whether an unsigned operation stayed in range, checked by re-deriving it
/// at full width and comparing against the truncated answer.
fn fits_unsigned(
    kind: IntKind,
    left: u128,
    right: u128,
    raw: u128,
    wrapping: &impl Fn(u128, u128) -> u128,
) -> bool {
    if kind.bits >= 128 {
        return true;
    }
    let wide = wrapping(left, right);
    wide == kind.truncate(raw) && wide <= kind.mask()
}
