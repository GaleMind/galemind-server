all: format test 
	cargo build

test:
	cargo test

e2e-test:
	./tests/run_e2e_tests.sh

test-unit:
	cargo test --workspace --exclude e2e-tests

#coverage:
#	cargo tarpaulin --config tarpaulin.toml --fail-under 80

format:
	cargo fmt

run:
	cargo run -p galemind start
docker-build:
	docker build -t galemind-server .

docker-run:
	docker run --rm -p 8080:8080 galemind-server