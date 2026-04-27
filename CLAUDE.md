# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project measuring carbon footprint and energy consumption during LLM fine-tuning. Uses LoRA/PEFT for efficient fine-tuning with energy monitoring via CodeCarbon, Zeus, and Prometheus/Grafana stack.

## Commands

```bash
# Setup
uv venv
uv sync

# Generate Docker infrastructure for a model
# First edit params.yml with model configuration, then:
uv run main.py

# Run fine-tuning inside Docker (from generated model directory)
docker compose up --build
```

## Architecture

### Scaffolding System (`main.py`)
The `main.py` script generates per-model Docker infrastructure from templates:
- Reads `params.yml` for model name, size, and container versions
- Creates `{MODEL_NAME}/{SIZE}/` directory with Dockerfile, docker-compose.yml, and empty training script
- Uses Python string.Template substitution on `template.Dockerfile` and `docker-compose-template.yml`

### Model Training Scripts
Each model directory (e.g., `gemma-3/12b/`, `Ministral-3/14B/`) contains a standalone Python script that:
1. Self-installs dependencies via `uv pip install` at runtime
2. Authenticates with HuggingFace and W&B using `.env` credentials
3. Tracks energy across phases (dataset, load_model, fine_tuning, evaluation) using:
   - **CodeCarbon**: CPU/RAM power tracking to `code_carbon_*/emissions.csv`
   - **Zeus**: GPU energy monitoring via `ZeusMonitor`
   - **Prometheus**: Real-time metrics exposed on port 8000

### Docker Infrastructure
- ROCm-based PyTorch containers for AMD GPUs
- Prometheus + Grafana + Node Exporter + AMD Device Metrics Exporter + Kepler for energy observability
- Metrics stored in `{MODEL_NAME}/{SIZE}/metrics/`

### Key Patterns
- Training uses `SFTTrainer` from TRL with PEFT/LoRA adapters
- All scripts define `begin_phase()`/`end_phase()` to wrap major operations
- Power timeseries logged to CSV and exposed as Prometheus gauges
- Models pushed to HuggingFace Hub after training

## Environment Variables (.env)

Required variables:
- `WANDB`: Weights & Biases API key
- `HF_TOKEN`: HuggingFace token with write access

## params.yml Configuration

```yaml
MODEL_NAME: gemma-3      # Model family directory name
SIZE: 12b                # Model size subdirectory
PYTORCH_VERSION: ...     # ROCm PyTorch image tag
PROMETHEUS_VERSION: ...
GRAFANA_VERSION: ...
NODE_EXPORTER_VERSION: ...
AMD_DME_VERSION: ...
KEPLER_VERSION: ...
```
