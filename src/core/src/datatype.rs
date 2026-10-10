//! Tensor data types defined by the Predict Protocol v2.

use crate::error::InferError;
use std::fmt;
use std::str::FromStr;

/// The element type of a tensor, as named by the Predict Protocol v2.
///
/// The string form is the canonical wire representation and is case-sensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Fp16,
    Fp32,
    Fp64,
    Bytes,
    Bf16,
}

impl DataType {
    /// The canonical protocol name, e.g. `"FP32"`.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::Uint8 => "UINT8",
            Self::Uint16 => "UINT16",
            Self::Uint32 => "UINT32",
            Self::Uint64 => "UINT64",
            Self::Int8 => "INT8",
            Self::Int16 => "INT16",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::Fp16 => "FP16",
            Self::Fp32 => "FP32",
            Self::Fp64 => "FP64",
            Self::Bytes => "BYTES",
            Self::Bf16 => "BF16",
        }
    }

    /// Size of one element in bytes, or `None` for the variable-width `BYTES` type.
    pub const fn byte_width(&self) -> Option<usize> {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 => Some(1),
            Self::Uint16 | Self::Int16 | Self::Fp16 | Self::Bf16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Fp32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Fp64 => Some(8),
            Self::Bytes => None,
        }
    }

    /// `true` for types that have no portable JSON representation and must
    /// therefore travel as raw bytes (see the "Tensor Data" section of the spec).
    pub const fn is_raw_only(&self) -> bool {
        matches!(self, Self::Fp16 | Self::Bf16)
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DataType {
    type Err = InferError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "BOOL" => Ok(Self::Bool),
            "UINT8" => Ok(Self::Uint8),
            "UINT16" => Ok(Self::Uint16),
            "UINT32" => Ok(Self::Uint32),
            "UINT64" => Ok(Self::Uint64),
            "INT8" => Ok(Self::Int8),
            "INT16" => Ok(Self::Int16),
            "INT32" => Ok(Self::Int32),
            "INT64" => Ok(Self::Int64),
            "FP16" => Ok(Self::Fp16),
            "FP32" => Ok(Self::Fp32),
            "FP64" => Ok(Self::Fp64),
            "BYTES" => Ok(Self::Bytes),
            "BF16" => Ok(Self::Bf16),
            other => Err(InferError::InvalidRequest(format!(
                "unknown datatype '{other}'"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_datatype_round_trips_through_its_protocol_name() {
        for dt in [
            DataType::Bool,
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Fp16,
            DataType::Fp32,
            DataType::Fp64,
            DataType::Bytes,
            DataType::Bf16,
        ] {
            assert_eq!(DataType::from_str(dt.as_str()).unwrap(), dt);
        }
    }

    #[test]
    fn datatype_names_are_case_sensitive() {
        assert!(DataType::from_str("fp32").is_err());
    }

    #[test]
    fn bytes_has_no_fixed_width() {
        assert_eq!(DataType::Bytes.byte_width(), None);
        assert_eq!(DataType::Fp32.byte_width(), Some(4));
    }
}
