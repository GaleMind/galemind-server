//! Encoding and decoding tensor contents as JSON.
//!
//! The specification gives tensor data as a JSON array whose elements may be
//! integers, floating-point numbers, strings or booleans, either in the
//! tensor's natural nested form or flattened. Decoding is therefore driven by
//! the declared `datatype`, never by what the JSON happens to look like: `[1, 2]`
//! is an `INT32` tensor or an `FP32` tensor depending on what the request says
//! it is.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use galemind_core::{DataType, InferError, TensorData};
use half::{bf16, f16};
use serde_json::{json, Number, Value};

type Result<T> = std::result::Result<T, InferError>;

/// Flattens a possibly nested JSON array into its elements, in row-major order.
fn flatten(name: &str, value: &Value, out: &mut Vec<Value>) -> Result<()> {
    match value {
        Value::Array(items) => {
            for item in items {
                flatten(name, item, out)?;
            }
            Ok(())
        }
        Value::Null => Err(InferError::InvalidRequest(format!(
            "tensor '{name}' holds a null element"
        ))),
        scalar => {
            out.push(scalar.clone());
            Ok(())
        }
    }
}

fn wrong_element(name: &str, datatype: DataType, value: &Value) -> InferError {
    InferError::InvalidRequest(format!(
        "tensor '{name}' is {datatype} but holds the element {value}"
    ))
}

/// Reads an integer element, rejecting anything that is not exactly one.
fn integer<T>(name: &str, datatype: DataType, value: &Value) -> Result<T>
where
    T: TryFrom<i64>,
{
    let number = value
        .as_i64()
        .ok_or_else(|| wrong_element(name, datatype, value))?;
    T::try_from(number).map_err(|_| {
        InferError::InvalidRequest(format!(
            "tensor '{name}' holds {number}, which is outside the range of {datatype}"
        ))
    })
}

/// Reads an unsigned element.
fn unsigned<T>(name: &str, datatype: DataType, value: &Value) -> Result<T>
where
    T: TryFrom<u64>,
{
    let number = value
        .as_u64()
        .ok_or_else(|| wrong_element(name, datatype, value))?;
    T::try_from(number).map_err(|_| {
        InferError::InvalidRequest(format!(
            "tensor '{name}' holds {number}, which is outside the range of {datatype}"
        ))
    })
}

fn float(name: &str, datatype: DataType, value: &Value) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| wrong_element(name, datatype, value))
}

/// The parameter naming how a `BYTES` tensor's strings are encoded.
///
/// JSON has no unambiguous way to carry bytes that are not text, and guessing
/// from the string's shape would silently corrupt data that happens to look
/// like base64. So the encoding is stated instead, using the server-specific
/// parameters the specification allows: `"binary_encoding": "base64"`. Text
/// tensors need nothing and stay readable.
pub const BINARY_ENCODING_PARAMETER: &str = "binary_encoding";

/// How the strings of a `BYTES` tensor are to be read or written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytesEncoding {
    /// Each string is its own UTF-8 bytes.
    Text,
    /// Each string is base64 of the element's bytes.
    Base64,
}

impl BytesEncoding {
    /// Reads the encoding a request declared, defaulting to text.
    pub fn from_parameters(parameters: Option<&crate::wire::JsonParameters>) -> Result<Self> {
        match parameters.and_then(|p| p.get(BINARY_ENCODING_PARAMETER)) {
            None => Ok(Self::Text),
            Some(Value::String(name)) if name == "base64" => Ok(Self::Base64),
            Some(Value::String(name)) if name == "text" => Ok(Self::Text),
            Some(other) => Err(InferError::InvalidRequest(format!(
                "{BINARY_ENCODING_PARAMETER} is {other}; it must be \"text\" or \"base64\""
            ))),
        }
    }
}

/// Reads a `BYTES` element in the declared encoding.
fn bytes(name: &str, value: &Value, encoding: BytesEncoding) -> Result<Vec<u8>> {
    match value {
        Value::String(text) => match encoding {
            BytesEncoding::Text => Ok(text.as_bytes().to_vec()),
            BytesEncoding::Base64 => BASE64.decode(text).map_err(|e| {
                InferError::InvalidRequest(format!(
                    "tensor '{name}' declares base64 contents but holds {text:?}: {e}"
                ))
            }),
        },
        other => Err(wrong_element(name, DataType::Bytes, other)),
    }
}

/// Decodes a `data` value into tensor contents of the declared datatype.
pub fn decode(
    name: &str,
    datatype: DataType,
    data: &Value,
    encoding: BytesEncoding,
) -> Result<TensorData> {
    let mut elements = Vec::new();
    flatten(name, data, &mut elements)?;

    Ok(match datatype {
        DataType::Bool => TensorData::Bool(
            elements
                .iter()
                .map(|v| v.as_bool().ok_or_else(|| wrong_element(name, datatype, v)))
                .collect::<Result<_>>()?,
        ),
        DataType::Uint8 => TensorData::Uint8(
            elements
                .iter()
                .map(|v| unsigned(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Uint16 => TensorData::Uint16(
            elements
                .iter()
                .map(|v| unsigned(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Uint32 => TensorData::Uint32(
            elements
                .iter()
                .map(|v| unsigned(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Uint64 => TensorData::Uint64(
            elements
                .iter()
                .map(|v| unsigned(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Int8 => TensorData::Int8(
            elements
                .iter()
                .map(|v| integer(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Int16 => TensorData::Int16(
            elements
                .iter()
                .map(|v| integer(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Int32 => TensorData::Int32(
            elements
                .iter()
                .map(|v| integer(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Int64 => TensorData::Int64(
            elements
                .iter()
                .map(|v| integer(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        DataType::Fp32 => TensorData::Fp32(
            elements
                .iter()
                .map(|v| float(name, datatype, v).map(|f| f as f32))
                .collect::<Result<_>>()?,
        ),
        DataType::Fp64 => TensorData::Fp64(
            elements
                .iter()
                .map(|v| float(name, datatype, v))
                .collect::<Result<_>>()?,
        ),
        // The specification notes that 16-bit floats have no standard JSON
        // representation. They are accepted as ordinary numbers and converted,
        // which loses precision beyond the format's own; gRPC raw contents
        // carry them exactly.
        DataType::Fp16 => TensorData::Fp16(
            elements
                .iter()
                .map(|v| float(name, datatype, v).map(|f| f16::from_f64(f)))
                .collect::<Result<_>>()?,
        ),
        DataType::Bf16 => TensorData::Bf16(
            elements
                .iter()
                .map(|v| float(name, datatype, v).map(|f| bf16::from_f64(f)))
                .collect::<Result<_>>()?,
        ),
        DataType::Bytes => TensorData::Bytes(
            elements
                .iter()
                .map(|v| bytes(name, v, encoding))
                .collect::<Result<_>>()?,
        ),
    })
}

/// Encodes tensor contents as a flat JSON array, reporting how `BYTES`
/// elements were written.
///
/// The flattened form is used because the specification allows it and it is
/// both smaller and cheaper to produce than the nested form. Byte strings are
/// written as text when every element is valid UTF-8, and as base64 otherwise;
/// the returned encoding is what the response declares in that output's
/// parameters.
pub fn encode(data: &TensorData) -> (Value, BytesEncoding) {
    fn number(value: f64) -> Value {
        // JSON has no way to write a non-finite number, so those become null,
        // which is the only representable choice.
        Number::from_f64(value).map_or(Value::Null, Value::Number)
    }

    let encoding = match data {
        TensorData::Bytes(v) if v.iter().any(|e| std::str::from_utf8(e).is_err()) => {
            BytesEncoding::Base64
        }
        _ => BytesEncoding::Text,
    };

    let value = match data {
        TensorData::Bool(v) => json!(v),
        TensorData::Uint8(v) => json!(v),
        TensorData::Uint16(v) => json!(v),
        TensorData::Uint32(v) => json!(v),
        TensorData::Uint64(v) => json!(v),
        TensorData::Int8(v) => json!(v),
        TensorData::Int16(v) => json!(v),
        TensorData::Int32(v) => json!(v),
        TensorData::Int64(v) => json!(v),
        TensorData::Fp32(v) => Value::Array(v.iter().map(|f| number(*f as f64)).collect()),
        TensorData::Fp64(v) => Value::Array(v.iter().map(|f| number(*f)).collect()),
        TensorData::Fp16(v) => Value::Array(v.iter().map(|f| number(f.to_f64())).collect()),
        TensorData::Bf16(v) => Value::Array(v.iter().map(|f| number(f.to_f64())).collect()),
        TensorData::Bytes(v) => Value::Array(
            v.iter()
                .map(|bytes| match encoding {
                    BytesEncoding::Text => Value::String(
                        String::from_utf8(bytes.clone()).expect("checked to be valid UTF-8"),
                    ),
                    BytesEncoding::Base64 => Value::String(BASE64.encode(bytes)),
                })
                .collect(),
        ),
    };

    (value, encoding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declared_datatype_decides_how_data_is_read() {
        let flat = json!([1, 2]);
        assert_eq!(
            decode("t", DataType::Int32, &flat, BytesEncoding::Text).unwrap(),
            TensorData::Int32(vec![1, 2])
        );
        assert_eq!(
            decode("t", DataType::Fp32, &flat, BytesEncoding::Text).unwrap(),
            TensorData::Fp32(vec![1.0, 2.0])
        );
        assert_eq!(
            decode("t", DataType::Uint64, &flat, BytesEncoding::Text).unwrap(),
            TensorData::Uint64(vec![1, 2])
        );
    }

    #[test]
    fn nested_and_flat_data_decode_identically() {
        let nested = decode(
            "t",
            DataType::Fp32,
            &json!([[1, 2], [4, 5]]),
            BytesEncoding::Text,
        )
        .unwrap();
        let flat = decode(
            "t",
            DataType::Fp32,
            &json!([1, 2, 4, 5]),
            BytesEncoding::Text,
        )
        .unwrap();
        assert_eq!(nested, flat);
        assert_eq!(nested, TensorData::Fp32(vec![1.0, 2.0, 4.0, 5.0]));
    }

    #[test]
    fn booleans_are_read_as_booleans_and_not_as_numbers() {
        assert_eq!(
            decode(
                "t",
                DataType::Bool,
                &json!([true, false]),
                BytesEncoding::Text
            )
            .unwrap(),
            TensorData::Bool(vec![true, false])
        );
        let err = decode("t", DataType::Bool, &json!([1]), BytesEncoding::Text).unwrap_err();
        assert!(err.to_string().contains("is BOOL"), "{err}");
    }

    #[test]
    fn an_out_of_range_integer_is_rejected_for_the_narrow_types() {
        let err = decode("t", DataType::Int8, &json!([200]), BytesEncoding::Text).unwrap_err();
        assert!(
            err.to_string().contains("outside the range of INT8"),
            "{err}"
        );

        let err = decode("t", DataType::Uint8, &json!([-1]), BytesEncoding::Text).unwrap_err();
        assert!(err.to_string().contains("is UINT8"), "{err}");
    }

    #[test]
    fn a_fractional_value_is_not_an_integer() {
        let err = decode("t", DataType::Int32, &json!([1.5]), BytesEncoding::Text).unwrap_err();
        assert!(err.to_string().contains("is INT32"), "{err}");
    }

    #[test]
    fn a_null_element_is_rejected() {
        assert!(decode("t", DataType::Fp32, &json!([1, null]), BytesEncoding::Text).is_err());
    }

    #[test]
    fn sixteen_bit_floats_are_accepted_as_ordinary_numbers() {
        assert_eq!(
            decode(
                "t",
                DataType::Fp16,
                &json!([1.5, -0.25]),
                BytesEncoding::Text
            )
            .unwrap(),
            TensorData::Fp16(vec![f16::from_f32(1.5), f16::from_f32(-0.25)])
        );
        assert_eq!(
            decode("t", DataType::Bf16, &json!([2.5]), BytesEncoding::Text).unwrap(),
            TensorData::Bf16(vec![bf16::from_f32(2.5)])
        );
    }

    #[test]
    fn text_bytes_round_trip_as_readable_json_strings() {
        let data = TensorData::Bytes(vec![b"hello".to_vec(), b"world".to_vec()]);
        let (encoded, encoding) = encode(&data);
        assert_eq!(encoded, json!(["hello", "world"]));
        assert_eq!(encoding, BytesEncoding::Text);
        assert_eq!(
            decode("t", DataType::Bytes, &encoded, encoding).unwrap(),
            data
        );
    }

    #[test]
    fn bytes_that_are_not_text_are_declared_as_base64_and_round_trip() {
        let data = TensorData::Bytes(vec![vec![0xff, 0x00, 0xfe]]);
        let (encoded, encoding) = encode(&data);
        assert_eq!(encoding, BytesEncoding::Base64);
        assert_eq!(encoded, json!(["/wD+"]));
        assert_eq!(
            decode("t", DataType::Bytes, &encoded, encoding).unwrap(),
            data
        );
    }

    #[test]
    fn a_base64_string_is_taken_literally_unless_the_request_says_otherwise() {
        // "/wD+" is valid base64, but in a text tensor it means exactly those
        // four characters. Guessing here would silently corrupt the data.
        let as_text = decode("t", DataType::Bytes, &json!(["/wD+"]), BytesEncoding::Text).unwrap();
        assert_eq!(as_text, TensorData::Bytes(vec![b"/wD+".to_vec()]));

        let as_binary = decode(
            "t",
            DataType::Bytes,
            &json!(["/wD+"]),
            BytesEncoding::Base64,
        )
        .unwrap();
        assert_eq!(as_binary, TensorData::Bytes(vec![vec![0xff, 0x00, 0xfe]]));
    }

    #[test]
    fn a_malformed_base64_element_is_rejected() {
        let err = decode(
            "t",
            DataType::Bytes,
            &json!(["not base64!"]),
            BytesEncoding::Base64,
        )
        .unwrap_err();
        assert!(err.is_client_error(), "{err}");
    }

    #[test]
    fn the_encoding_parameter_is_read_from_a_requests_parameters() {
        use crate::wire::JsonParameters;

        assert_eq!(
            BytesEncoding::from_parameters(None).unwrap(),
            BytesEncoding::Text
        );

        let base64: JsonParameters = [(
            BINARY_ENCODING_PARAMETER.to_string(),
            Value::String("base64".to_string()),
        )]
        .into_iter()
        .collect();
        assert_eq!(
            BytesEncoding::from_parameters(Some(&base64)).unwrap(),
            BytesEncoding::Base64
        );

        let nonsense: JsonParameters = [(BINARY_ENCODING_PARAMETER.to_string(), Value::Bool(true))]
            .into_iter()
            .collect();
        assert!(BytesEncoding::from_parameters(Some(&nonsense)).is_err());
    }

    #[test]
    fn every_fixed_width_datatype_round_trips_through_json() {
        let cases = vec![
            TensorData::Bool(vec![true, false]),
            TensorData::Uint8(vec![0, 255]),
            TensorData::Uint16(vec![65_535]),
            TensorData::Uint32(vec![u32::MAX]),
            TensorData::Uint64(vec![u64::MAX]),
            TensorData::Int8(vec![-128, 127]),
            TensorData::Int16(vec![-32_768]),
            TensorData::Int32(vec![i32::MIN, i32::MAX]),
            TensorData::Int64(vec![i64::MIN, i64::MAX]),
            TensorData::Fp32(vec![1.5, -2.25]),
            TensorData::Fp64(vec![0.5, -0.125]),
            TensorData::Fp16(vec![f16::from_f32(0.5)]),
            TensorData::Bf16(vec![bf16::from_f32(0.5)]),
        ];
        for data in cases {
            let datatype = data.datatype();
            let back = decode("t", datatype, &encode(&data).0, BytesEncoding::Text).unwrap();
            assert_eq!(back, data, "round trip failed for {datatype}");
        }
    }

    #[test]
    fn a_non_finite_float_is_encoded_as_null_because_json_cannot_hold_it() {
        assert_eq!(
            encode(&TensorData::Fp32(vec![f32::NAN, f32::INFINITY])).0,
            json!([null, null])
        );
    }

    #[test]
    fn a_scalar_data_value_is_a_single_element() {
        assert_eq!(
            decode("t", DataType::Int32, &json!(7), BytesEncoding::Text).unwrap(),
            TensorData::Int32(vec![7])
        );
    }
}
