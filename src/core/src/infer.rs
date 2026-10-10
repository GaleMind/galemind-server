//! Inference requests and responses, independent of transport.

use crate::error::InferError;
use crate::model::{ModelMetadata, ModelRef};
use crate::tensor::Tensor;
use std::collections::BTreeMap;

/// A protocol parameter value. The Predict Protocol v2 allows exactly these
/// three kinds for gRPC `InferParameter`; JSON numbers are narrowed to `Int64`.
#[derive(Debug, Clone, PartialEq)]
pub enum Parameter {
    Bool(bool),
    Int64(i64),
    String(String),
}

/// Zero or more named parameters. Ordered so that encoded output is stable.
pub type Parameters = BTreeMap<String, Parameter>;

/// An output the client asked for by name.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestedOutput {
    pub name: String,
    pub parameters: Parameters,
}

/// One inference request, already decoded into domain tensors.
#[derive(Debug, Clone)]
pub struct InferRequest {
    pub model: ModelRef,
    pub id: Option<String>,
    pub parameters: Parameters,
    pub inputs: Vec<Tensor>,
    /// Empty means "return every output the model produces".
    pub requested_outputs: Vec<RequestedOutput>,
}

impl InferRequest {
    /// Checks the request against the model's metadata before it reaches a
    /// backend: inputs must exist, be well formed, carry the right datatype and
    /// agree with the model's fixed dimensions.
    pub fn validate_against(&self, metadata: &ModelMetadata) -> Result<(), InferError> {
        if self.inputs.is_empty() {
            return Err(InferError::InvalidRequest(
                "request carries no input tensors".to_string(),
            ));
        }

        let mut seen = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            if seen.contains(&input.name) {
                return Err(InferError::InvalidRequest(format!(
                    "input '{}' appears more than once",
                    input.name
                )));
            }
            seen.push(input.name.clone());

            input.validate()?;

            let expected = metadata.input(&input.name).ok_or_else(|| {
                InferError::InvalidRequest(format!(
                    "model '{}' has no input named '{}'",
                    metadata.name, input.name
                ))
            })?;

            if expected.datatype != input.datatype() {
                return Err(InferError::InvalidRequest(format!(
                    "input '{}' has datatype {} but model expects {}",
                    input.name,
                    input.datatype(),
                    expected.datatype
                )));
            }

            if expected.shape.len() != input.shape.len() {
                return Err(InferError::InvalidRequest(format!(
                    "input '{}' has rank {} but model expects rank {}",
                    input.name,
                    input.shape.len(),
                    expected.shape.len()
                )));
            }

            for (axis, (actual, declared)) in
                input.shape.iter().zip(expected.shape.iter()).enumerate()
            {
                if *declared >= 0 && declared != actual {
                    return Err(InferError::InvalidRequest(format!(
                        "input '{}' has {actual} elements on axis {axis} but model expects {declared}",
                        input.name
                    )));
                }
            }
        }

        // The first missing input is the error; a request missing two is
        // still one request to fix.
        if let Some(missing) = metadata
            .inputs
            .iter()
            .map(|t| &t.name)
            .find(|name| !seen.contains(*name))
        {
            return Err(InferError::InvalidRequest(format!(
                "request is missing required input '{missing}'"
            )));
        }

        for requested in &self.requested_outputs {
            if metadata.output(&requested.name).is_none() {
                return Err(InferError::InvalidRequest(format!(
                    "model '{}' has no output named '{}'",
                    metadata.name, requested.name
                )));
            }
        }

        Ok(())
    }

    /// Size of the leading dimension shared by every input, which is how many
    /// independent rows this request contributes to a batch.
    pub fn batch_rows(&self) -> Result<usize, InferError> {
        let mut rows = None;
        for input in &self.inputs {
            let input_rows = input.rows()?;
            match rows {
                None => rows = Some(input_rows),
                Some(existing) if existing != input_rows => {
                    return Err(InferError::InvalidRequest(format!(
                        "input '{}' has batch dimension {input_rows}, expected {existing}",
                        input.name
                    )));
                }
                Some(_) => {}
            }
        }
        rows.ok_or_else(|| {
            InferError::InvalidRequest("request carries no input tensors".to_string())
        })
    }
}

/// One inference result.
#[derive(Debug, Clone)]
pub struct InferResponse {
    pub model_name: String,
    pub model_version: String,
    pub id: Option<String>,
    pub parameters: Parameters,
    pub outputs: Vec<Tensor>,
}

impl InferResponse {
    /// Keeps only the outputs the client asked for, preserving the requested
    /// order. An empty selection returns everything.
    pub fn select_outputs(mut self, requested: &[RequestedOutput]) -> Self {
        if requested.is_empty() {
            return self;
        }
        let mut selected = Vec::with_capacity(requested.len());
        for want in requested {
            if let Some(pos) = self.outputs.iter().position(|o| o.name == want.name) {
                selected.push(self.outputs.remove(pos));
            }
        }
        self.outputs = selected;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::DataType;
    use crate::model::TensorMetadata;
    use crate::tensor::TensorData;

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            name: "building_energy".into(),
            versions: vec!["1".into()],
            platform: "onnx_onnxv1".into(),
            inputs: vec![TensorMetadata {
                name: "features".into(),
                datatype: DataType::Fp32,
                shape: vec![-1, 2],
            }],
            outputs: vec![TensorMetadata {
                name: "kwh".into(),
                datatype: DataType::Fp32,
                shape: vec![-1, 1],
            }],
            max_batch_size: Some(16),
        }
    }

    fn request(inputs: Vec<Tensor>) -> InferRequest {
        InferRequest {
            model: ModelRef::latest("building_energy"),
            id: None,
            parameters: Parameters::new(),
            inputs,
            requested_outputs: Vec::new(),
        }
    }

    fn features(shape: Vec<i64>, data: Vec<f32>) -> Tensor {
        Tensor::new("features", shape, TensorData::Fp32(data))
    }

    #[test]
    fn a_well_formed_request_passes_validation() {
        let req = request(vec![features(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])]);
        req.validate_against(&metadata()).unwrap();
        assert_eq!(req.batch_rows().unwrap(), 2);
    }

    #[test]
    fn a_fixed_dimension_must_match_exactly() {
        let req = request(vec![features(vec![1, 3], vec![1.0, 2.0, 3.0])]);
        let err = req.validate_against(&metadata()).unwrap_err();
        assert!(err.to_string().contains("axis 1"), "{err}");
    }

    #[test]
    fn the_declared_datatype_must_match_the_model() {
        let req = request(vec![Tensor::new(
            "features",
            vec![1, 2],
            TensorData::Fp64(vec![1.0, 2.0]),
        )]);
        let err = req.validate_against(&metadata()).unwrap_err();
        assert!(err.to_string().contains("datatype FP64"), "{err}");
    }

    #[test]
    fn an_unknown_input_name_is_rejected() {
        let req = request(vec![Tensor::new(
            "nope",
            vec![1, 2],
            TensorData::Fp32(vec![1.0, 2.0]),
        )]);
        assert!(req.validate_against(&metadata()).is_err());
    }

    #[test]
    fn a_missing_input_is_rejected() {
        let err = request(Vec::new())
            .validate_against(&metadata())
            .unwrap_err();
        assert!(err.to_string().contains("no input tensors"), "{err}");
    }

    #[test]
    fn a_duplicated_input_is_rejected() {
        let req = request(vec![
            features(vec![1, 2], vec![1.0, 2.0]),
            features(vec![1, 2], vec![3.0, 4.0]),
        ]);
        let err = req.validate_against(&metadata()).unwrap_err();
        assert!(err.to_string().contains("more than once"), "{err}");
    }

    #[test]
    fn an_unknown_requested_output_is_rejected() {
        let mut req = request(vec![features(vec![1, 2], vec![1.0, 2.0])]);
        req.requested_outputs = vec![RequestedOutput {
            name: "ghost".into(),
            parameters: Parameters::new(),
        }];
        assert!(req.validate_against(&metadata()).is_err());
    }

    #[test]
    fn select_outputs_keeps_the_requested_order_and_drops_the_rest() {
        let response = InferResponse {
            model_name: "m".into(),
            model_version: "1".into(),
            id: None,
            parameters: Parameters::new(),
            outputs: vec![
                Tensor::new("a", vec![1], TensorData::Int64(vec![1])),
                Tensor::new("b", vec![1], TensorData::Int64(vec![2])),
                Tensor::new("c", vec![1], TensorData::Int64(vec![3])),
            ],
        };
        let requested = vec![
            RequestedOutput {
                name: "c".into(),
                parameters: Parameters::new(),
            },
            RequestedOutput {
                name: "a".into(),
                parameters: Parameters::new(),
            },
        ];
        let names: Vec<_> = response
            .select_outputs(&requested)
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect();
        assert_eq!(names, vec!["c", "a"]);
    }
}
