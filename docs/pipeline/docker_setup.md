# Docker Setup Guide

## Prerequisites
- Docker Engine 20.10.0+
- Docker Compose 2.0.0+

## Quick Start
```bash
# Build and start
docker-compose up -d --build

# Verify
docker-compose ps
```

## Key Commands
- **Run a command**: `docker-compose exec pluto <command>`
- **View logs**: `docker-compose logs -f`
- **Stop**: `docker-compose down`

## Volume Mounts
- `./:/workspace` - Project files
- `~/.cache/huggingface` - Model cache

## Development Workflow
1. Make code changes locally
2. Test in container:
   ```bash
   docker-compose exec -T pluto python scripts/prepare_lora_data.py
   ```
3. View results in `data/processed/`

[Next: Advanced Usage →](advanced_usage.md)
