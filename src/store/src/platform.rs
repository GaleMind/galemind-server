//! Mapping model files onto Predict Protocol v2 platform strings.

use galemind_core::PLATFORM_ONNX;
use std::path::Path;

/// File name to platform, in the order candidates are tried.
///
/// Serving a new runtime means adding a row here and registering its backend
/// factory; nothing else in the store changes.
const PLATFORMS: &[(&str, &str)] = &[
    ("model.onnx", PLATFORM_ONNX),
    ("model.plan", "tensorrt_plan"),
    ("model.graphdef", "tensorflow_graphdef"),
    ("model.savedmodel", "tensorflow_savedmodel"),
    ("model.pt", "pytorch_torchscript"),
    ("model.netdef", "caffe2_netdef"),
];

/// The file names a model version directory may hold, most preferred first.
pub fn candidate_file_names() -> impl Iterator<Item = &'static str> {
    PLATFORMS.iter().map(|(file, _)| *file)
}

/// The platform served by a given model file, if it is one we recognise.
pub fn platform_for_file(path: &Path) -> Option<&'static str> {
    let name = path.file_name()?.to_str()?;
    PLATFORMS
        .iter()
        .find(|(file, _)| *file == name)
        .map(|(_, platform)| *platform)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn an_onnx_model_file_maps_to_the_onnx_platform() {
        let path = PathBuf::from("/models/building_energy/1/model.onnx");
        assert_eq!(platform_for_file(&path), Some(PLATFORM_ONNX));
    }

    #[test]
    fn an_unknown_file_has_no_platform() {
        assert_eq!(platform_for_file(&PathBuf::from("notes.txt")), None);
    }

    #[test]
    fn onnx_is_tried_first() {
        assert_eq!(candidate_file_names().next(), Some("model.onnx"));
    }
}
