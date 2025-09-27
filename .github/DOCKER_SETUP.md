# Docker Hub Setup Guide

This guide explains how to configure Docker Hub integration for automatic image building and pushing.

## Prerequisites

1. **Docker Hub Account**: Create an account at [hub.docker.com](https://hub.docker.com)
2. **Docker Hub Repository**: Create a repository named `galemind-server`

## GitHub Secrets Configuration

Add the following secrets to your GitHub repository:

### Required Secrets

1. **DOCKER_USERNAME**: Your Docker Hub username
   - Go to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `DOCKER_USERNAME`
   - Value: Your Docker Hub username

2. **DOCKER_PASSWORD**: Your Docker Hub access token
   - Go to Docker Hub → Account Settings → Security
   - Click "New Access Token"
   - Name it "GitHub Actions"
   - Copy the generated token
   - Add as secret: `DOCKER_PASSWORD`

## Update Image Repository

1. Replace `yourusername` in the following files with your actual Docker Hub username:
   - `infra/helm/galemind-server/values.yaml`
   - `infra/environments/dev/values.yaml`
   - `infra/environments/staging/values.yaml`
   - `infra/environments/production/values.yaml`
   - `Dockerfile` (source label)

2. Update the repository URLs from:
   ```yaml
   repository: docker.io/yourusername/galemind-server
   ```
   To:
   ```yaml
   repository: docker.io/YOUR_ACTUAL_USERNAME/galemind-server
   ```

## Image Tagging Strategy

The workflow automatically creates the following tags:

### Branch-based Tags
- `main` branch → `latest` and `dev` tags
- `develop` branch → `staging` tag
- Pull requests → `pr-<number>` tag

### Release Tags
- `v1.0.0` → `v1.0.0`, `v1.0`, `v1`, `latest`
- `v1.2.3` → `v1.2.3`, `v1.2`, `v1`

## Workflow Triggers

The Docker build workflow runs on:

1. **Push to main branch**: Creates `latest` and `dev` tags
2. **Push to develop branch**: Creates `staging` tag
3. **Version tags** (v*): Creates versioned tags
4. **Pull requests**: Builds but doesn't push (for testing)

## Security Features

- **Vulnerability Scanning**: Images are scanned with Trivy
- **Multi-platform**: Builds for `linux/amd64` and `linux/arm64`
- **Layer Caching**: Uses GitHub Actions cache for faster builds
- **Non-root User**: Container runs as non-root user `galemind`

## Manual Build and Push

To manually build and push:

```bash
# Build the image
docker build -t your-username/galemind-server:latest .

# Push to Docker Hub
docker push your-username/galemind-server:latest
```

## Troubleshooting

### Authentication Issues
- Verify Docker Hub credentials in GitHub secrets
- Ensure access token has write permissions
- Check repository name matches Docker Hub repository

### Build Failures
- Check GitHub Actions logs for detailed error messages
- Verify Dockerfile syntax
- Ensure all dependencies are available

### Permission Issues
- Verify Docker Hub repository is public or accessible
- Check access token permissions
- Ensure GitHub repository has necessary permissions

## Next Steps

1. Replace placeholder usernames with your actual Docker Hub username
2. Add the required secrets to your GitHub repository
3. Push to main branch to trigger the first build
4. Verify the image appears in your Docker Hub repository