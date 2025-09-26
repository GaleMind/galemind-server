all: format test 
	cargo build

test:
	cargo test

#coverage:
#	cargo tarpaulin --config tarpaulin.toml --fail-under 80

format:
	cargo fmt

run:
	cargo run -p galemind start
