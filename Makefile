all: format test 
	cargo build

test:
	cargo test

#coverage:
#	cargo tarpaulin --config tarpaulin.toml --fail-under 80

format:
	cargo fmt

run:
	@if [ -f .env ]; then \
		set -a && . ./.env && set +a; \
	fi && \
	cargo run -p galemind start
docker-build:
	$(eval TIMESTAMP := $(shell date +%Y%m%d%H%M))
	docker build -t galemind-server:$(TIMESTAMP) .

docker-run:
	$(eval TIMESTAMP := $(shell date +%Y%m%d%H%M))
	docker run --rm -p 8080:8080 galemind-server:$(TIMESTAMP)

docker-push:
	$(eval TIMESTAMP := $(shell date +%Y%m%d%H%M))
	docker push galemind-server:$(TIMESTAMP)