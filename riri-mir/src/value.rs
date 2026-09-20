//! What an interpreted MIR local can hold.
//!
//! Scalars carry their own width and signedness rather than leaning on the
//! type of whatever holds them, because MIR arithmetic is width-dependent:
//! `CheckedBinaryOp` reports overflow, and overflow is only defined once you
//! know how many bits there were. Bits are kept zero-extended in a `u128` and
//! interpreted on demand, which is the same shape rustc uses.

use std::fmt;

/// The width and signedness of an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntKind {
    pub bits: u32,
    pub signed: bool,
}

impl IntKind {
    pub fn new(bits: u32, signed: bool) -> Self {
        IntKind { bits, signed }
    }

    /// Mask covering the value's width.
    pub fn mask(self) -> u128 {
        if self.bits >= 128 {
            u128::MAX
        } else {
            (1u128 << self.bits) - 1
        }
    }

    /// Truncates to this width, discarding anything above it.
    pub fn truncate(self, raw: u128) -> u128 {
        raw & self.mask()
    }

    /// Reads the bits as a signed value, extending the sign bit.
    pub fn as_signed(self, raw: u128) -> i128 {
        let raw = self.truncate(raw);
        if self.bits >= 128 {
            return raw as i128;
        }
        let sign = 1u128 << (self.bits - 1);
        if raw & sign != 0 {
            (raw | !self.mask()) as i128
        } else {
            raw as i128
        }
    }

    /// Whether `raw` still fits once truncated, which is how overflow is
    /// decided for both signednesses.
    pub fn fits(self, raw: i128) -> bool {
        let truncated = self.as_signed(raw as u128);
        if self.signed {
            truncated == raw
        } else {
            raw >= 0 && self.truncate(raw as u128) == raw as u128
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A local that has been declared but not assigned. Reading one is a bug
    /// in the interpreted program, or in this interpreter.
    Uninit,
    Int { raw: u128, kind: IntKind },
    Bool(bool),
    /// Tuples and structs, including the pair `CheckedBinaryOp` produces.
    Aggregate(Vec<Value>),
}

impl Value {
    pub fn int(raw: u128, kind: IntKind) -> Self {
        Value::Int {
            raw: kind.truncate(raw),
            kind,
        }
    }

    pub fn as_int(&self) -> Result<(u128, IntKind), String> {
        match self {
            Value::Int { raw, kind } => Ok((*raw, *kind)),
            other => Err(format!("expected an integer, found {other:?}")),
        }
    }

    pub fn as_bool(&self) -> Result<bool, String> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(format!("expected a bool, found {other:?}")),
        }
    }

    /// The value as it would select a `SwitchInt` arm.
    pub fn discriminant(&self) -> Result<u128, String> {
        match self {
            Value::Bool(b) => Ok(*b as u128),
            Value::Int { raw, .. } => Ok(*raw),
            other => Err(format!("cannot switch on {other:?}")),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Uninit => write!(f, "uninit"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int { raw, kind } => {
                if kind.signed {
                    write!(f, "{}_i{}", kind.as_signed(*raw), kind.bits)
                } else {
                    write!(f, "{raw}_u{}", kind.bits)
                }
            }
            Value::Aggregate(parts) => {
                write!(f, "(")?;
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{part}")?;
                }
                write!(f, ")")
            }
        }
    }
}
