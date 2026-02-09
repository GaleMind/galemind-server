#!/bin/bash

set -e

echo "🚀 Starting E2E Tests for GaleMind Server"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed or not in PATH"
    exit 1
fi

# Navigate to the project root
cd "$(dirname "$0")/.."

print_status "Starting services with docker-compose..."

# Start services in detached mode
docker-compose -f docker-compose.dev.yml up -d

print_status "Waiting for services to be ready..."

# Wait for services to be healthy
timeout=300  # 5 minutes
elapsed=0
interval=5

while [ $elapsed -lt $timeout ]; do
    if docker-compose -f docker-compose.dev.yml ps | grep -q "healthy"; then
        print_status "Services are healthy!"
        break
    fi

    if [ $elapsed -eq 0 ]; then
        print_warning "Waiting for services to become healthy..."
    fi

    sleep $interval
    elapsed=$((elapsed + interval))

    if [ $elapsed -ge $timeout ]; then
        print_error "Services did not become healthy within $timeout seconds"
        print_status "Service status:"
        docker-compose -f docker-compose.dev.yml ps

        print_status "Service logs:"
        docker-compose -f docker-compose.dev.yml logs

        # Cleanup
        docker-compose -f docker-compose.dev.yml down
        exit 1
    fi
done

print_status "Running E2E tests..."

# Run the tests
cd tests
if cargo test --verbose; then
    print_status "✅ All E2E tests passed!"
    test_result=0
else
    print_error "❌ Some E2E tests failed!"
    test_result=1
fi

# Cleanup
cd ..
print_status "Cleaning up services..."
docker-compose -f docker-compose.dev.yml down

if [ $test_result -eq 0 ]; then
    print_status "🎉 E2E test run completed successfully!"
else
    print_error "💥 E2E test run failed!"
fi

exit $test_result