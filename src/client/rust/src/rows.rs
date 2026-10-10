//! Turning rows of numbers into a tensor the model accepts.

use galemind_core::{DataType, InferError, ModelMetadata, Tensor, TensorData, TensorMetadata};

type Result<T> = std::result::Result<T, InferError>;

/// Parses a comma-separated row such as `1800,30,-4,18,22,1,0,40`.
pub fn parse_row(row: &str) -> Result<Vec<f64>> {
    row.split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .map(|field| {
            field
                .parse::<f64>()
                .map_err(|_| InferError::InvalidRequest(format!("'{field}' is not a number")))
        })
        .collect()
}

/// The single input a model takes, so a caller need not name it.
///
/// Models with several inputs cannot be addressed this way, and say so.
pub fn sole_input(metadata: &ModelMetadata) -> Result<&TensorMetadata> {
    match metadata.inputs.as_slice() {
        [only] => Ok(only),
        [] => Err(InferError::InvalidRequest(format!(
            "model '{}' declares no inputs",
            metadata.name
        ))),
        many => Err(InferError::InvalidRequest(format!(
            "model '{}' takes {} inputs ({}), so each must be given explicitly",
            metadata.name,
            many.len(),
            many.iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// How many values one row of an input holds, from the model's own shape.
pub fn row_width(input: &TensorMetadata) -> Result<usize> {
    let mut width = 1usize;
    for dim in input.shape.iter().skip(1) {
        let dim = usize::try_from(*dim).map_err(|_| {
            InferError::InvalidRequest(format!(
                "input '{}' has a variable dimension past the first, \
                 so rows cannot be inferred",
                input.name
            ))
        })?;
        width *= dim;
    }
    Ok(width)
}

/// Builds a batched input tensor from rows of numbers.
///
/// The rows are checked against the model's declared width before anything is
/// sent, which turns a typo into a message rather than a server-side error.
pub fn rows_to_tensor(input: &TensorMetadata, rows: &[Vec<f64>]) -> Result<Tensor> {
    if rows.is_empty() {
        return Err(InferError::InvalidRequest(
            "no input rows were given".to_string(),
        ));
    }

    let width = row_width(input)?;
    for (index, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(InferError::InvalidRequest(format!(
                "row {} holds {} values but input '{}' takes {width}",
                index + 1,
                row.len(),
                input.name
            )));
        }
    }

    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let data = match input.datatype {
        DataType::Fp32 => TensorData::Fp32(flat.iter().map(|v| *v as f32).collect()),
        DataType::Fp64 => TensorData::Fp64(flat),
        DataType::Int32 => TensorData::Int32(flat.iter().map(|v| *v as i32).collect()),
        DataType::Int64 => TensorData::Int64(flat.iter().map(|v| *v as i64).collect()),
        other => {
            return Err(InferError::InvalidRequest(format!(
                "input '{}' is {other}; rows of numbers can only fill \
                 FP32, FP64, INT32 or INT64 inputs",
                input.name
            )))
        }
    };

    let mut shape = vec![rows.len() as i64];
    shape.extend(input.shape.iter().skip(1).copied());
    Ok(Tensor::new(&input.name, shape, data))
}

/// Generates reproducible rows for load testing.
///
/// The values come from a small deterministic generator rather than a random
/// source, so a run can be repeated exactly from its seed.
pub fn synthesize_rows(count: usize, width: usize, seed: u64) -> Vec<Vec<f64>> {
    // The seed is mixed before use. Feeding it to xorshift directly would make
    // neighbouring seeds produce near-identical streams, which is exactly what
    // someone sweeping seeds across load-test runs would hit.
    let mut state = splitmix64(seed);
    let mut next = || {
        // xorshift64*, enough for feeding a load test.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let value = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 11) as f64 / (1u64 << 53) as f64
    };

    (0..count)
        .map(|_| (0..width).map(|_| next() * 100.0).collect())
        .collect()
}

/// Spreads a seed across the whole 64-bit range, so that close seeds give
/// unrelated streams. Never returns zero, which xorshift cannot escape.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(datatype: DataType, shape: Vec<i64>) -> TensorMetadata {
        TensorMetadata {
            name: "features".to_string(),
            datatype,
            shape,
        }
    }

    fn metadata(inputs: Vec<TensorMetadata>) -> ModelMetadata {
        ModelMetadata {
            name: "energy".to_string(),
            versions: vec!["1".to_string()],
            platform: "onnx_onnxv1".to_string(),
            inputs,
            outputs: Vec::new(),
            max_batch_size: None,
        }
    }

    #[test]
    fn a_row_of_numbers_is_parsed_with_whitespace_allowed() {
        assert_eq!(parse_row(" 1, 2.5 ,-3 ").unwrap(), vec![1.0, 2.5, -3.0]);
    }

    #[test]
    fn a_row_holding_something_other_than_a_number_is_refused() {
        let err = parse_row("1,two").unwrap_err();
        assert!(err.to_string().contains("'two' is not a number"), "{err}");
    }

    #[test]
    fn the_single_input_of_a_model_is_found_automatically() {
        let metadata = metadata(vec![input(DataType::Fp32, vec![-1, 8])]);
        assert_eq!(sole_input(&metadata).unwrap().name, "features");
    }

    #[test]
    fn a_model_with_several_inputs_says_they_must_be_named() {
        let metadata = metadata(vec![
            input(DataType::Fp32, vec![-1, 8]),
            TensorMetadata {
                name: "mask".to_string(),
                datatype: DataType::Bool,
                shape: vec![-1, 8],
            },
        ]);
        let err = sole_input(&metadata).unwrap_err();
        assert!(err.to_string().contains("features, mask"), "{err}");
    }

    #[test]
    fn rows_become_a_batched_tensor_of_the_models_datatype() {
        let input = input(DataType::Fp32, vec![-1, 3]);
        let tensor = rows_to_tensor(&input, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();

        assert_eq!(tensor.name, "features");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.datatype(), DataType::Fp32);
        tensor.validate().unwrap();
    }

    #[test]
    fn an_integer_input_is_filled_from_the_same_rows() {
        let input = input(DataType::Int64, vec![-1, 2]);
        let tensor = rows_to_tensor(&input, &[vec![1.0, 2.0]]).unwrap();
        assert_eq!(tensor.data, TensorData::Int64(vec![1, 2]));
    }

    #[test]
    fn a_row_of_the_wrong_width_is_refused_before_anything_is_sent() {
        let input = input(DataType::Fp32, vec![-1, 8]);
        let err = rows_to_tensor(&input, &[vec![1.0, 2.0]]).unwrap_err();
        assert!(err.to_string().contains("takes 8"), "{err}");
    }

    #[test]
    fn no_rows_at_all_is_refused() {
        assert!(rows_to_tensor(&input(DataType::Fp32, vec![-1, 8]), &[]).is_err());
    }

    #[test]
    fn an_input_that_rows_cannot_fill_says_so() {
        let err = rows_to_tensor(&input(DataType::Bytes, vec![-1, 1]), &[vec![1.0]]).unwrap_err();
        assert!(err.to_string().contains("BYTES"), "{err}");
    }

    #[test]
    fn synthetic_rows_are_reproducible_from_their_seed() {
        let first = synthesize_rows(4, 8, 42);
        assert_eq!(first.len(), 4);
        assert_eq!(first[0].len(), 8);
        assert_eq!(first, synthesize_rows(4, 8, 42));
        assert!(first.iter().flatten().all(|v| (0.0..100.0).contains(v)));
    }

    #[test]
    fn neighbouring_seeds_give_unrelated_rows() {
        // Sweeping seeds across runs must actually vary the data.
        for seed in 0..8u64 {
            assert_ne!(
                synthesize_rows(2, 4, seed),
                synthesize_rows(2, 4, seed + 1),
                "seeds {seed} and {} produced the same rows",
                seed + 1
            );
        }
    }
}
