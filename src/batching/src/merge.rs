//! Merging requests into one batched call and splitting the result back.
//!
//! Tensor contents are held flattened in row-major order, so a batch is a plain
//! concatenation along the leading dimension and a split is a plain range copy.
//! No transposition or re-ordering is involved, which is why this is pure
//! domain code with no backend dependency.

use galemind_core::{InferError, InferRequest, InferResponse, Parameters, Tensor};

type Result<T> = std::result::Result<T, InferError>;

/// Concatenates several requests into one.
///
/// Every request must carry the same input names; the merged tensors keep the
/// order of the first request. Returns the merged request together with the row
/// count each original contributed, which [`split_response`] needs.
pub fn merge_requests(requests: &[InferRequest]) -> Result<(InferRequest, Vec<usize>)> {
    let first = requests
        .first()
        .ok_or_else(|| InferError::Internal("cannot merge an empty batch".to_string()))?;

    if requests.len() == 1 {
        return Ok((first.clone(), vec![first.batch_rows()?]));
    }

    let mut rows_per_request = Vec::with_capacity(requests.len());
    let mut merged: Vec<Tensor> = first
        .inputs
        .iter()
        .map(|input| Tensor::new(&input.name, input.shape.clone(), input.data.clone()))
        .collect();
    rows_per_request.push(first.batch_rows()?);

    for request in &requests[1..] {
        let rows = request.batch_rows()?;
        if request.inputs.len() != merged.len() {
            return Err(InferError::Internal(format!(
                "cannot batch requests with {} and {} inputs",
                merged.len(),
                request.inputs.len()
            )));
        }

        for target in merged.iter_mut() {
            let source = request
                .inputs
                .iter()
                .find(|input| input.name == target.name)
                .ok_or_else(|| {
                    InferError::Internal(format!(
                        "cannot batch requests: one lacks input '{}'",
                        target.name
                    ))
                })?;

            if source.row_len()? != target.row_len()? {
                return Err(InferError::Internal(format!(
                    "cannot batch input '{}': rows of {} and {} elements",
                    target.name,
                    target.row_len()?,
                    source.row_len()?
                )));
            }

            target.data.append(source.data.clone())?;
            target.shape[0] += rows as i64;
        }
        rows_per_request.push(rows);
    }

    Ok((
        InferRequest {
            model: first.model.clone(),
            // The merged request is an internal artefact: it carries no single
            // client's id, and per-request parameters stay with their response.
            id: None,
            parameters: Parameters::new(),
            inputs: merged,
            requested_outputs: Vec::new(),
        },
        rows_per_request,
    ))
}

/// Splits a batched response into one response per original request.
///
/// `rows_per_request` comes from [`merge_requests`] and must sum to the rows the
/// response holds.
pub fn split_response(
    response: &InferResponse,
    rows_per_request: &[usize],
) -> Result<Vec<InferResponse>> {
    let total_rows: usize = rows_per_request.iter().sum();
    if rows_per_request.len() == 1 {
        return Ok(vec![response.clone()]);
    }

    let mut split: Vec<InferResponse> = rows_per_request
        .iter()
        .map(|_| InferResponse {
            model_name: response.model_name.clone(),
            model_version: response.model_version.clone(),
            id: None,
            parameters: response.parameters.clone(),
            outputs: Vec::with_capacity(response.outputs.len()),
        })
        .collect();

    for output in &response.outputs {
        if total_rows == 0 {
            return Err(InferError::Internal(
                "cannot split a batch of zero rows".to_string(),
            ));
        }
        let elements = output.data.len();
        if elements % total_rows != 0 {
            return Err(InferError::Internal(format!(
                "output '{}' holds {elements} elements, which {total_rows} rows do not divide",
                output.name
            )));
        }
        let row_len = elements / total_rows;

        let mut offset = 0usize;
        for (index, rows) in rows_per_request.iter().enumerate() {
            let mut shape = output.shape.clone();
            if shape.is_empty() {
                return Err(InferError::Internal(format!(
                    "output '{}' is a scalar and cannot be split per request",
                    output.name
                )));
            }
            shape[0] = *rows as i64;

            split[index].outputs.push(Tensor::new(
                &output.name,
                shape,
                output.data.slice(offset, rows * row_len)?,
            ));
            offset += rows * row_len;
        }
    }

    Ok(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use galemind_core::{ModelRef, RequestedOutput, TensorData};

    fn request(id: &str, rows: usize) -> InferRequest {
        let values: Vec<f32> = (0..rows * 2).map(|v| v as f32).collect();
        InferRequest {
            model: ModelRef::latest("energy"),
            id: Some(id.to_string()),
            parameters: Parameters::new(),
            inputs: vec![Tensor::new(
                "features",
                vec![rows as i64, 2],
                TensorData::Fp32(values),
            )],
            requested_outputs: vec![RequestedOutput {
                name: "kwh".to_string(),
                parameters: Parameters::new(),
            }],
        }
    }

    fn response(rows: usize, row_len: usize) -> InferResponse {
        let values: Vec<f32> = (0..rows * row_len).map(|v| v as f32).collect();
        InferResponse {
            model_name: "energy".to_string(),
            model_version: "1".to_string(),
            id: None,
            parameters: Parameters::new(),
            outputs: vec![Tensor::new(
                "kwh",
                vec![rows as i64, row_len as i64],
                TensorData::Fp32(values),
            )],
        }
    }

    #[test]
    fn merging_concatenates_rows_and_remembers_how_many_each_request_gave() {
        let batch = vec![request("a", 1), request("b", 3), request("c", 2)];
        let (merged, rows) = merge_requests(&batch).unwrap();

        assert_eq!(rows, vec![1, 3, 2]);
        assert_eq!(merged.inputs.len(), 1);
        assert_eq!(merged.inputs[0].shape, vec![6, 2]);
        assert_eq!(merged.inputs[0].data.len(), 12);
        merged.inputs[0].validate().unwrap();
        assert_eq!(
            merged.id, None,
            "the merged request is not any one client's"
        );
    }

    #[test]
    fn merging_a_single_request_leaves_it_untouched() {
        let (merged, rows) = merge_requests(&[request("a", 4)]).unwrap();
        assert_eq!(rows, vec![4]);
        assert_eq!(merged.id.as_deref(), Some("a"));
    }

    #[test]
    fn merging_nothing_is_an_internal_error() {
        assert!(merge_requests(&[]).is_err());
    }

    #[test]
    fn requests_with_different_row_widths_cannot_be_merged() {
        let mut odd = request("b", 1);
        odd.inputs[0] = Tensor::new("features", vec![1, 3], TensorData::Fp32(vec![0.0; 3]));
        let err = merge_requests(&[request("a", 1), odd]).unwrap_err();
        assert!(err.to_string().contains("rows of 2 and 3"), "{err}");
    }

    #[test]
    fn requests_for_different_inputs_cannot_be_merged() {
        let mut other = request("b", 1);
        other.inputs[0].name = "other".to_string();
        assert!(merge_requests(&[request("a", 1), other]).is_err());
    }

    #[test]
    fn splitting_returns_each_requests_own_rows() {
        let parts = split_response(&response(6, 3), &[1, 3, 2]).unwrap();

        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].outputs[0].shape, vec![1, 3]);
        assert_eq!(parts[1].outputs[0].shape, vec![3, 3]);
        assert_eq!(parts[2].outputs[0].shape, vec![2, 3]);

        let TensorData::Fp32(first) = &parts[0].outputs[0].data else {
            panic!("expected FP32");
        };
        assert_eq!(first, &[0.0, 1.0, 2.0]);

        let TensorData::Fp32(last) = &parts[2].outputs[0].data else {
            panic!("expected FP32");
        };
        assert_eq!(last, &[12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
        for part in &parts {
            part.outputs[0].validate().unwrap();
        }
    }

    #[test]
    fn splitting_handles_a_rank_one_output() {
        let batched = InferResponse {
            outputs: vec![Tensor::new(
                "band",
                vec![4],
                TensorData::Int64(vec![0, 1, 2, 1]),
            )],
            ..response(4, 1)
        };
        let parts = split_response(&batched, &[3, 1]).unwrap();
        assert_eq!(parts[0].outputs[0].shape, vec![3]);
        assert_eq!(parts[0].outputs[0].data, TensorData::Int64(vec![0, 1, 2]));
        assert_eq!(parts[1].outputs[0].data, TensorData::Int64(vec![1]));
    }

    #[test]
    fn a_response_whose_rows_do_not_divide_is_rejected() {
        let err = split_response(&response(5, 1), &[2, 2]).unwrap_err();
        assert!(err.to_string().contains("do not divide"), "{err}");
    }

    #[test]
    fn merge_then_split_restores_every_requests_shape() {
        let batch = vec![request("a", 2), request("b", 1), request("c", 5)];
        let (merged, rows) = merge_requests(&batch).unwrap();
        let total: usize = rows.iter().sum();
        assert_eq!(merged.inputs[0].shape[0], total as i64);

        let parts = split_response(&response(total, 1), &rows).unwrap();
        for (part, request) in parts.iter().zip(batch.iter()) {
            assert_eq!(part.outputs[0].shape[0], request.inputs[0].shape[0]);
        }
    }
}
