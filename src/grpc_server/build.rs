fn main() {
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["proto/grpc_predict_v2.proto"], &["proto"])
        .expect("the Predict Protocol v2 service definition should compile");
}
