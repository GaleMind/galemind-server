# Galemind Server Infrastructure

This directory contains Kubernetes deployment configurations using Helm charts for the Galemind Server across three environments: development, staging, and production.

## Structure

```
infra/
├── helm/
│   └── galemind-server/
│       ├── Chart.yaml              # Helm chart metadata and dependencies
│       ├── values.yaml             # Default values
│       └── templates/
│           ├── deployment.yaml     # Kubernetes deployment with GPU/CPU specs
│           ├── service.yaml        # Kubernetes service
│           ├── serviceaccount.yaml # Service account
│           ├── ingress.yaml        # Ingress configuration
│           └── _helpers.tpl        # Helm template helpers
└── environments/
    ├── dev/
    │   └── values.yaml             # Development environment values
    ├── staging/
    │   └── values.yaml             # Staging environment values
    └── production/
        └── values.yaml             # Production environment values
```

## Features

- **Multi-environment support**: Separate configurations for dev, staging, and production
- **MLflow integration**: Deployed as a separate service with automatic service discovery
- **HashiCorp Vault**: Secret management with automatic injection
- **GPU/CPU specifications**: Resource limits and requests in deployment metadata
- **Auto-scaling**: Horizontal Pod Autoscaler for staging and production
- **Ingress**: Traffic routing with TLS support
- **Persistent storage**: MLflow data persistence across pod restarts

## Deployment

### Prerequisites

1. Kubernetes cluster with GPU nodes
2. Helm 3.x installed
3. HashiCorp Vault configured
4. NVIDIA device plugin for GPU support

### Deploy to Development

```bash
helm upgrade --install galemind-server-dev ./infra/helm/galemind-server \
  -f ./infra/environments/dev/values.yaml \
  --namespace galemind-dev \
  --create-namespace
```

### Deploy to Staging

```bash
helm upgrade --install galemind-server-staging ./infra/helm/galemind-server \
  -f ./infra/environments/staging/values.yaml \
  --namespace galemind-staging \
  --create-namespace
```

### Deploy to Production

```bash
helm upgrade --install galemind-server-prod ./infra/helm/galemind-server \
  -f ./infra/environments/production/values.yaml \
  --namespace galemind-production \
  --create-namespace
```

## Configuration

### GPU Resources

Each environment is configured with appropriate GPU allocations:
- **Dev**: 1 GPU (nvidia.com/gpu: 1)
- **Staging**: 1 GPU with auto-scaling
- **Production**: 2 GPUs with advanced node affinity

### Vault Secrets

The deployment uses HashiCorp Vault for secret management with automatic injection:
- Secrets are injected as environment variables
- Each environment has its own vault role and secret path
- Common secrets: `database_url`, `api_key`

### MLflow Integration

MLflow is deployed as a separate service within the same Helm chart:
- **Service Discovery**: Galemind automatically connects to MLflow via Kubernetes DNS
- **Standard Docker Image**: Uses Python 3.9 slim with MLflow installed
- **Persistent Storage**: MLflow data persists across pod restarts
- **Environment Variables**:
  - `MLFLOW_TRACKING_URI`: Automatically set to MLflow service endpoint
  - `MLFLOW_TRACKING_HOST`: MLflow service name
  - `MLFLOW_TRACKING_PORT`: MLflow service port (5000)

#### MLflow Service Names by Environment:
- **Dev**: `galemind-server-dev-mlflow:5000`
- **Staging**: `galemind-server-staging-mlflow:5000`
- **Production**: `galemind-server-prod-mlflow:5000`

## Environment Differences

| Feature | Development | Staging | Production |
|---------|-------------|---------|------------|
| **Galemind Server** | | | |
| Replicas | 1 | 2-5 (auto-scale) | 3-10 (auto-scale) |
| CPU Limit | 500m | 1000m | 2000m |
| Memory Limit | 1Gi | 2Gi | 4Gi |
| GPU Count | 1 | 1 | 2 |
| Log Level | debug | info | warn |
| TLS | No | Staging cert | Production cert |
| **MLflow** | | | |
| Replicas | 1 | 1 | 2 |
| CPU Limit | 250m | 500m | 1000m |
| Memory Limit | 512Mi | 1Gi | 2Gi |
| Storage | 5Gi | 20Gi | 50Gi |
| Database | File-based | File-based | PostgreSQL |

## Monitoring

Health checks are configured:
- **Liveness probe**: `/health` endpoint
- **Readiness probe**: `/ready` endpoint
- **Initial delay**: 30s for liveness, 5s for readiness