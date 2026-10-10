//! Translation between domain tensors and ONNX Runtime values.

use galemind_core::{DataType, InferError, Tensor, TensorData, TensorMetadata};
use ort::value::{DynValue, Value, ValueType};
use ort::value::{Tensor as OrtTensor, TensorElementType};

type Result<T> = std::result::Result<T, InferError>;

/// ONNX Runtime errors carry the type of the call that produced them; this
/// flattens any of them into the domain error.
fn backend_err<T>(e: ort::Error<T>) -> InferError {
    InferError::Backend(e.to_string())
}

/// Maps an ONNX Runtime element type onto a protocol datatype.
///
/// ONNX string tensors carry the protocol's `BYTES` type, which is how the
/// Predict Protocol v2 represents variable-length elements.
pub fn datatype_from_ort(ty: TensorElementType) -> Result<DataType> {
    Ok(match ty {
        TensorElementType::Bool => DataType::Bool,
        TensorElementType::Uint8 => DataType::Uint8,
        TensorElementType::Uint16 => DataType::Uint16,
        TensorElementType::Uint32 => DataType::Uint32,
        TensorElementType::Uint64 => DataType::Uint64,
        TensorElementType::Int8 => DataType::Int8,
        TensorElementType::Int16 => DataType::Int16,
        TensorElementType::Int32 => DataType::Int32,
        TensorElementType::Int64 => DataType::Int64,
        TensorElementType::Float16 => DataType::Fp16,
        TensorElementType::Float32 => DataType::Fp32,
        TensorElementType::Float64 => DataType::Fp64,
        TensorElementType::Bfloat16 => DataType::Bf16,
        TensorElementType::String => DataType::Bytes,
        other => {
            return Err(InferError::Unsupported(format!(
                "ONNX element type {other:?} has no Predict Protocol v2 datatype"
            )))
        }
    })
}

/// Describes one model input or output from its ONNX Runtime type info.
pub fn tensor_metadata(name: &str, ty: &ValueType) -> Result<TensorMetadata> {
    match ty {
        ValueType::Tensor { ty, shape, .. } => Ok(TensorMetadata {
            name: name.to_string(),
            datatype: datatype_from_ort(*ty)?,
            shape: shape.iter().copied().collect(),
        }),
        other => Err(InferError::Unsupported(format!(
            "model input/output '{name}' is a {other:?}, only tensors are supported"
        ))),
    }
}

/// Builds an ONNX Runtime value from a domain tensor.
pub fn to_ort_value(tensor: &Tensor) -> Result<DynValue> {
    let shape = tensor.shape.clone();

    macro_rules! owned {
        ($values:expr) => {
            OrtTensor::from_array((shape, $values.clone()))
                .map_err(backend_err)?
                .into_dyn()
        };
    }

    Ok(match &tensor.data {
        TensorData::Bool(v) => owned!(v),
        TensorData::Uint8(v) => owned!(v),
        TensorData::Uint16(v) => owned!(v),
        TensorData::Uint32(v) => owned!(v),
        TensorData::Uint64(v) => owned!(v),
        TensorData::Int8(v) => owned!(v),
        TensorData::Int16(v) => owned!(v),
        TensorData::Int32(v) => owned!(v),
        TensorData::Int64(v) => owned!(v),
        TensorData::Fp16(v) => owned!(v),
        TensorData::Bf16(v) => owned!(v),
        TensorData::Fp32(v) => owned!(v),
        TensorData::Fp64(v) => owned!(v),
        TensorData::Bytes(v) => {
            let strings = v
                .iter()
                .map(|bytes| {
                    String::from_utf8(bytes.clone()).map_err(|_| {
                        InferError::InvalidRequest(format!(
                            "BYTES input '{}' holds non-UTF-8 data, which ONNX string tensors cannot carry",
                            tensor.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            OrtTensor::from_string_array((shape, strings.as_slice()))
                .map_err(backend_err)?
                .into_dyn()
        }
    })
}

/// Reads an ONNX Runtime value back into a domain tensor.
pub fn from_ort_value(name: &str, value: &Value) -> Result<Tensor> {
    let datatype = match value.dtype() {
        ValueType::Tensor { ty, .. } => datatype_from_ort(*ty)?,
        other => {
            return Err(InferError::Unsupported(format!(
                "model output '{name}' is a {other:?}, only tensors are supported"
            )))
        }
    };

    macro_rules! extract {
        ($t:ty, $variant:ident) => {{
            let (shape, values) = value.try_extract_tensor::<$t>().map_err(backend_err)?;
            (
                shape.iter().copied().collect::<Vec<i64>>(),
                TensorData::$variant(values.to_vec()),
            )
        }};
    }

    let (shape, data) = match datatype {
        DataType::Bool => extract!(bool, Bool),
        DataType::Uint8 => extract!(u8, Uint8),
        DataType::Uint16 => extract!(u16, Uint16),
        DataType::Uint32 => extract!(u32, Uint32),
        DataType::Uint64 => extract!(u64, Uint64),
        DataType::Int8 => extract!(i8, Int8),
        DataType::Int16 => extract!(i16, Int16),
        DataType::Int32 => extract!(i32, Int32),
        DataType::Int64 => extract!(i64, Int64),
        DataType::Fp16 => extract!(half::f16, Fp16),
        DataType::Bf16 => extract!(half::bf16, Bf16),
        DataType::Fp32 => extract!(f32, Fp32),
        DataType::Fp64 => extract!(f64, Fp64),
        DataType::Bytes => {
            let (shape, values) = value.try_extract_strings().map_err(backend_err)?;
            (
                shape.iter().copied().collect::<Vec<i64>>(),
                TensorData::Bytes(values.into_iter().map(String::into_bytes).collect()),
            )
        }
    };

    Ok(Tensor::new(name, shape, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_protocol_datatype_has_an_ort_counterpart() {
        let pairs = [
            (TensorElementType::Bool, DataType::Bool),
            (TensorElementType::Uint8, DataType::Uint8),
            (TensorElementType::Int64, DataType::Int64),
            (TensorElementType::Float16, DataType::Fp16),
            (TensorElementType::Bfloat16, DataType::Bf16),
            (TensorElementType::Float32, DataType::Fp32),
            (TensorElementType::Float64, DataType::Fp64),
            (TensorElementType::String, DataType::Bytes),
        ];
        for (ort_ty, expected) in pairs {
            assert_eq!(datatype_from_ort(ort_ty).unwrap(), expected);
        }
    }

    #[test]
    fn an_unrepresentable_element_type_is_rejected() {
        assert!(datatype_from_ort(TensorElementType::Complex64).is_err());
    }

    #[test]
    fn a_domain_tensor_round_trips_through_an_ort_value() {
        let tensor = Tensor::new(
            "features",
            vec![2, 2],
            TensorData::Fp32(vec![1.0, 2.0, 3.0, 4.0]),
        );
        let value = to_ort_value(&tensor).unwrap();
        assert_eq!(from_ort_value("features", &value).unwrap(), tensor);
    }

    #[test]
    fn a_bytes_tensor_round_trips_as_an_onnx_string_tensor() {
        let tensor = Tensor::new(
            "words",
            vec![2],
            TensorData::Bytes(vec![b"hello".to_vec(), b"world".to_vec()]),
        );
        let value = to_ort_value(&tensor).unwrap();
        assert_eq!(from_ort_value("words", &value).unwrap(), tensor);
    }

    #[test]
    fn non_utf8_bytes_input_is_reported_as_a_client_error() {
        let tensor = Tensor::new("words", vec![1], TensorData::Bytes(vec![vec![0xff, 0xfe]]));
        let err = to_ort_value(&tensor).unwrap_err();
        assert!(err.is_client_error(), "{err}");
    }
}
