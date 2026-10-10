//! Tensors and their element storage.
//!
//! Tensor contents are always held flattened in row-major order, exactly as the
//! Predict Protocol v2 requires on the wire, so no re-ordering is needed when a
//! tensor is encoded, decoded, concatenated into a batch or split back out.

use crate::datatype::DataType;
use crate::error::InferError;
use half::{bf16, f16};

type Result<T> = std::result::Result<T, InferError>;

/// A fixed-width element that can be moved to and from raw little-endian bytes.
trait RawCodec: Copy {
    const WIDTH: usize;
    fn write_le(self, out: &mut Vec<u8>);
    fn read_le(bytes: &[u8]) -> Self;
}

macro_rules! impl_raw_codec {
    ($($t:ty),* $(,)?) => {
        $(impl RawCodec for $t {
            const WIDTH: usize = std::mem::size_of::<$t>();
            fn write_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn read_le(bytes: &[u8]) -> Self {
                let mut buf = [0u8; std::mem::size_of::<$t>()];
                buf.copy_from_slice(bytes);
                <$t>::from_le_bytes(buf)
            }
        })*
    };
}

impl_raw_codec!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, f16, bf16);

impl RawCodec for bool {
    const WIDTH: usize = 1;
    fn write_le(self, out: &mut Vec<u8>) {
        out.push(u8::from(self));
    }
    fn read_le(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

macro_rules! define_tensor_data {
    ($($variant:ident($elem:ty) => $dt:ident,)*) => {
        /// The contents of a tensor, flattened in row-major order.
        ///
        /// `FP16`/`BF16` are kept as `half` types so that raw payloads survive a
        /// round trip bit-for-bit; `BYTES` holds one byte string per element.
        #[derive(Debug, Clone, PartialEq)]
        pub enum TensorData {
            $($variant(Vec<$elem>),)*
            Bytes(Vec<Vec<u8>>),
        }

        impl TensorData {
            /// The protocol datatype matching this storage.
            pub fn datatype(&self) -> DataType {
                match self {
                    $(Self::$variant(_) => DataType::$dt,)*
                    Self::Bytes(_) => DataType::Bytes,
                }
            }

            /// Number of elements held.
            pub fn len(&self) -> usize {
                match self {
                    $(Self::$variant(v) => v.len(),)*
                    Self::Bytes(v) => v.len(),
                }
            }

            /// An empty buffer of the given datatype.
            pub fn empty(datatype: DataType) -> Self {
                match datatype {
                    $(DataType::$dt => Self::$variant(Vec::new()),)*
                    DataType::Bytes => Self::Bytes(Vec::new()),
                }
            }

            /// Appends `other` onto `self`, used to concatenate a batch.
            pub fn append(&mut self, other: Self) -> Result<()> {
                match (self, other) {
                    $((Self::$variant(a), Self::$variant(b)) => { a.extend(b); Ok(()) })*
                    (Self::Bytes(a), Self::Bytes(b)) => { a.extend(b); Ok(()) }
                    (a, b) => Err(InferError::InvalidRequest(format!(
                        "cannot concatenate {} data with {} data",
                        a.datatype(),
                        b.datatype()
                    ))),
                }
            }

            /// Copies `len` elements starting at `start`, used to split a batched
            /// result back into per-request responses.
            pub fn slice(&self, start: usize, len: usize) -> Result<Self> {
                let end = start.checked_add(len).ok_or_else(|| {
                    InferError::Internal("tensor slice range overflows".to_string())
                })?;
                if end > self.len() {
                    return Err(InferError::Internal(format!(
                        "tensor slice {start}..{end} exceeds length {}",
                        self.len()
                    )));
                }
                Ok(match self {
                    $(Self::$variant(v) => Self::$variant(v[start..end].to_vec()),)*
                    Self::Bytes(v) => Self::Bytes(v[start..end].to_vec()),
                })
            }

            /// Encodes the contents as raw little-endian bytes. `BYTES` elements
            /// are each prefixed with their length as a 4-byte little-endian u32.
            pub fn to_raw(&self) -> Vec<u8> {
                let mut out = Vec::with_capacity(self.raw_size_hint());
                match self {
                    $(Self::$variant(v) => {
                        for e in v {
                            e.write_le(&mut out);
                        }
                    })*
                    Self::Bytes(v) => {
                        for e in v {
                            out.extend_from_slice(&(e.len() as u32).to_le_bytes());
                            out.extend_from_slice(e);
                        }
                    }
                }
                out
            }

            /// Decodes raw little-endian bytes holding exactly `count` elements.
            pub fn from_raw(datatype: DataType, bytes: &[u8], count: usize) -> Result<Self> {
                match datatype {
                    $(DataType::$dt => {
                        let width = <$elem as RawCodec>::WIDTH;
                        let expected = count * width;
                        if bytes.len() != expected {
                            return Err(InferError::InvalidRequest(format!(
                                "raw contents for {} tensor hold {} bytes, expected {expected}",
                                datatype,
                                bytes.len()
                            )));
                        }
                        Ok(Self::$variant(
                            bytes.chunks_exact(width).map(<$elem as RawCodec>::read_le).collect(),
                        ))
                    })*
                    DataType::Bytes => {
                        let mut out = Vec::with_capacity(count);
                        let mut cursor = 0usize;
                        while out.len() < count {
                            if cursor + 4 > bytes.len() {
                                return Err(InferError::InvalidRequest(
                                    "truncated length prefix in BYTES raw contents".to_string(),
                                ));
                            }
                            let len = u32::from_le_bytes(
                                bytes[cursor..cursor + 4].try_into().expect("4 bytes"),
                            ) as usize;
                            cursor += 4;
                            let end = cursor.checked_add(len).ok_or_else(|| {
                                InferError::InvalidRequest(
                                    "length prefix in BYTES raw contents overflows".to_string(),
                                )
                            })?;
                            if end > bytes.len() {
                                return Err(InferError::InvalidRequest(
                                    "truncated element in BYTES raw contents".to_string(),
                                ));
                            }
                            out.push(bytes[cursor..end].to_vec());
                            cursor = end;
                        }
                        if cursor != bytes.len() {
                            return Err(InferError::InvalidRequest(format!(
                                "BYTES raw contents hold {} trailing bytes after {count} elements",
                                bytes.len() - cursor
                            )));
                        }
                        Ok(Self::Bytes(out))
                    }
                }
            }

            fn raw_size_hint(&self) -> usize {
                match self.datatype().byte_width() {
                    Some(width) => self.len() * width,
                    None => match self {
                        Self::Bytes(v) => v.iter().map(|e| e.len() + 4).sum(),
                        _ => 0,
                    },
                }
            }
        }
    };
}

define_tensor_data! {
    Bool(bool) => Bool,
    Uint8(u8) => Uint8,
    Uint16(u16) => Uint16,
    Uint32(u32) => Uint32,
    Uint64(u64) => Uint64,
    Int8(i8) => Int8,
    Int16(i16) => Int16,
    Int32(i32) => Int32,
    Int64(i64) => Int64,
    Fp16(f16) => Fp16,
    Fp32(f32) => Fp32,
    Fp64(f64) => Fp64,
    Bf16(bf16) => Bf16,
}

impl TensorData {
    /// `true` when no elements are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named tensor: a shape plus its flattened contents.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: TensorData,
}

impl Tensor {
    pub fn new(name: impl Into<String>, shape: Vec<i64>, data: TensorData) -> Self {
        Self {
            name: name.into(),
            shape,
            data,
        }
    }

    pub fn datatype(&self) -> DataType {
        self.data.datatype()
    }

    /// Number of elements the shape describes.
    pub fn element_count(&self) -> Result<usize> {
        let mut count = 1usize;
        for dim in &self.shape {
            let dim = usize::try_from(*dim).map_err(|_| {
                InferError::InvalidRequest(format!(
                    "tensor '{}' has non-concrete dimension {dim}",
                    self.name
                ))
            })?;
            count = count.checked_mul(dim).ok_or_else(|| {
                InferError::InvalidRequest(format!("tensor '{}' shape overflows", self.name))
            })?;
        }
        Ok(count)
    }

    /// Number of elements in a single row, i.e. the product of all dimensions
    /// after the leading (batch) one.
    pub fn row_len(&self) -> Result<usize> {
        let mut count = 1usize;
        for dim in self.shape.iter().skip(1) {
            let dim = usize::try_from(*dim).map_err(|_| {
                InferError::InvalidRequest(format!(
                    "tensor '{}' has non-concrete dimension {dim}",
                    self.name
                ))
            })?;
            count = count.checked_mul(dim).ok_or_else(|| {
                InferError::InvalidRequest(format!("tensor '{}' shape overflows", self.name))
            })?;
        }
        Ok(count)
    }

    /// Size of the leading dimension, or 1 for a scalar-shaped tensor.
    pub fn rows(&self) -> Result<usize> {
        match self.shape.first() {
            Some(dim) => usize::try_from(*dim).map_err(|_| {
                InferError::InvalidRequest(format!(
                    "tensor '{}' has non-concrete batch dimension {dim}",
                    self.name
                ))
            }),
            None => Ok(1),
        }
    }

    /// Rejects a tensor whose contents do not match its declared shape.
    pub fn validate(&self) -> Result<()> {
        let expected = self.element_count()?;
        if expected != self.data.len() {
            return Err(InferError::InvalidRequest(format!(
                "tensor '{}' declares shape {:?} ({expected} elements) but holds {}",
                self.name,
                self.shape,
                self.data.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_round_trip_preserves_every_fixed_width_type() {
        let cases = vec![
            TensorData::Bool(vec![true, false, true]),
            TensorData::Uint8(vec![0, 7, 255]),
            TensorData::Uint16(vec![0, 65535]),
            TensorData::Uint32(vec![1, u32::MAX]),
            TensorData::Uint64(vec![1, u64::MAX]),
            TensorData::Int8(vec![-128, 127]),
            TensorData::Int16(vec![-1, 300]),
            TensorData::Int32(vec![i32::MIN, 0, i32::MAX]),
            TensorData::Int64(vec![i64::MIN, i64::MAX]),
            TensorData::Fp16(vec![f16::from_f32(1.5), f16::from_f32(-0.25)]),
            TensorData::Bf16(vec![bf16::from_f32(2.5)]),
            TensorData::Fp32(vec![1.0, -2.5, 3.25]),
            TensorData::Fp64(vec![f64::MIN, 0.5]),
        ];
        for data in cases {
            let raw = data.to_raw();
            let back = TensorData::from_raw(data.datatype(), &raw, data.len()).unwrap();
            assert_eq!(back, data, "round trip failed for {}", data.datatype());
        }
    }

    #[test]
    fn raw_round_trip_preserves_variable_width_bytes() {
        let data = TensorData::Bytes(vec![b"abc".to_vec(), Vec::new(), b"de".to_vec()]);
        let raw = data.to_raw();
        assert_eq!(raw.len(), 4 + 3 + 4 + 4 + 2);
        assert_eq!(
            TensorData::from_raw(DataType::Bytes, &raw, 3).unwrap(),
            data
        );
    }

    #[test]
    fn from_raw_rejects_a_payload_of_the_wrong_size() {
        let err = TensorData::from_raw(DataType::Fp32, &[0u8; 6], 2).unwrap_err();
        assert!(err.to_string().contains("expected 8"), "{err}");
    }

    #[test]
    fn from_raw_rejects_truncated_bytes_elements() {
        let mut raw = 9u32.to_le_bytes().to_vec();
        raw.extend_from_slice(b"ab");
        assert!(TensorData::from_raw(DataType::Bytes, &raw, 1).is_err());
    }

    #[test]
    fn append_concatenates_matching_types_and_rejects_mismatches() {
        let mut a = TensorData::Fp32(vec![1.0, 2.0]);
        a.append(TensorData::Fp32(vec![3.0])).unwrap();
        assert_eq!(a, TensorData::Fp32(vec![1.0, 2.0, 3.0]));
        assert!(a.append(TensorData::Int64(vec![1])).is_err());
    }

    #[test]
    fn slice_extracts_a_sub_range_and_rejects_out_of_bounds() {
        let data = TensorData::Int64(vec![1, 2, 3, 4]);
        assert_eq!(data.slice(1, 2).unwrap(), TensorData::Int64(vec![2, 3]));
        assert!(data.slice(3, 2).is_err());
    }

    #[test]
    fn validate_compares_shape_against_element_count() {
        let good = Tensor::new("t", vec![2, 2], TensorData::Fp32(vec![1.0, 2.0, 3.0, 4.0]));
        good.validate().unwrap();

        let bad = Tensor::new("t", vec![2, 2], TensorData::Fp32(vec![1.0]));
        let err = bad.validate().unwrap_err();
        assert!(err.to_string().contains("4 elements"), "{err}");
    }

    #[test]
    fn row_len_is_the_product_of_the_non_batch_dimensions() {
        let t = Tensor::new("t", vec![4, 2, 3], TensorData::empty(DataType::Fp32));
        assert_eq!(t.rows().unwrap(), 4);
        assert_eq!(t.row_len().unwrap(), 6);
    }

    #[test]
    fn a_negative_dimension_is_not_a_concrete_shape() {
        let t = Tensor::new("t", vec![-1, 4], TensorData::empty(DataType::Fp32));
        assert!(t.element_count().is_err());
    }
}
