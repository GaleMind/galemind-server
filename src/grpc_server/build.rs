/*
fn main() {
    tonic_build::compile_protos("proto/prediction/prediction.proto").unwrap();
}
*/

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/prediction/prediction.proto"], &["proto"])?;
    Ok(())
}

