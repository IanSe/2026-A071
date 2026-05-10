import sys
import os
import subprocess
import torch
 
# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
 
MODEL_NAME           = "Hugofernandez/Mistral-7B-v0.1-colab-sharded"
HUB_MODEL_ID         = "darmasrmz/bio-mistral-7b-quant"   # Change to your repo
PUSH_TO_HUB          = True
 
DATASET_NAME         = "bio-nlp-umass/bioinstruct"
TEST_SIZE            = 0.2
 
# Paths — Lightning AI persistent storage
OUTPUT_BASE          = "/teamspace/studios/this_studio/biomistral_finetune"
RESULTS_DIR          = os.path.join(OUTPUT_BASE, "results")
CARBON_DIR           = os.path.join(OUTPUT_BASE, "code_carbon")
SAVED_MODEL_DIR      = os.path.join(OUTPUT_BASE, "Biomistral_7B")
 
# ──────────────────────────────────────────────────────────────────────────────
# 1. INSTALL PACKAGES
# ──────────────────────────────────────────────────────────────────────────────
 
pkgs = [
    "python-dotenv", "transformers", "trl", "accelerate", "bitsandbytes",
    "datasets", "ninja", "peft", "codecarbon", "packaging", "zeus-ml",
    "wandb", "huggingface-hub", "tqdm", "evaluate", "bert_score",
    "prometheus-client", "pandas", "matplotlib",
]
 
 
def install_packages(packages):
    """Install packages, auto-detecting uv or falling back to pip."""
    installer = "pip"
    try:
        subprocess.check_call(
            ["uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        installer = "uv"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
 
    print(f"Installing packages via {installer}...")
    if installer == "uv":
        subprocess.check_call(["uv", "pip", "install"] + packages)
        subprocess.call(["uv", "pip", "uninstall", "amdsmi"])
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + packages
        )
        subprocess.call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "amdsmi"]
        )
    print("All packages installed!")
 
 
install_packages(pkgs)
 
# ──────────────────────────────────────────────────────────────────────────────
# 2. AUTH — Lightning AI Secrets or .env fallback
# ──────────────────────────────────────────────────────────────────────────────
 
from dotenv import load_dotenv
from huggingface_hub import login, auth_list
import wandb
 
load_dotenv()  # no-op on Lightning AI (no .env), harmless
wandb_key = os.getenv("WANDB")
hf_token  = os.getenv("HF_TOKEN")
 
if wandb_key:
    wandb.login(key=wandb_key)
else:
    print("⚠️  WANDB secret not found. W&B logging disabled.")
 
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    auth_list()
else:
    print("⚠️  HF_TOKEN secret not found. Hub push disabled.")
    PUSH_TO_HUB = False
 
# ──────────────────────────────────────────────────────────────────────────────
# 3. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
 
import json
from datetime import datetime, timezone
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging
from zeus.device import get_gpus
from zeus.monitor import ZeusMonitor
from prometheus_client import start_http_server, Gauge
 
import transformers
import trl as trl_mod
print(f"Transformers : {transformers.__version__}")
print(f"TRL          : {trl_mod.__version__}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
 
gpus = get_gpus()
print(f"Zeus GPUs    : {gpus}")
 
# ──────────────────────────────────────────────────────────────────────────────
# 4. HELPERS
# ──────────────────────────────────────────────────────────────────────────────
 
def format_sample(example):
    """Format BioInstruct sample into Mistral instruction template."""
    prompt = example["instruction"]
    if "input" in example and len(example["input"]) > 0:
        prompt += "\n" + example["input"]
    answer = example["output"]
    return {
        "text": f"<s>[INST]{prompt}[/INST]{answer}</s>"
    }
 
 
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 5. MONITORING SETUP (CodeCarbon + Prometheus + Zeus)
# ──────────────────────────────────────────────────────────────────────────────
 
os.makedirs(CARBON_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
 
log_name = "biomistral_7b_logs"
_logger = logging.getLogger(log_name)
_channel = logging.FileHandler(os.path.join(OUTPUT_BASE, log_name + ".log"))
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)
 
start_http_server(8000)
 
PHASE = Gauge("training_phase", "Current fine-tuning phase (1=active, 0=inactive)", ["phase"])
CPU_W = Gauge("training_cpu_power_watts", "CPU power from codecarbon (watts)")
RAM_W = Gauge("training_ram_power_watts", "RAM power from codecarbon, estimate (watts)")
GPU_W = Gauge("training_gpu_power_watts", "GPU power estimate (watts)")
CPU_E = Gauge("training_cpu_energy_kwh", "Cumulative CPU energy (kWh)")
RAM_E = Gauge("training_ram_energy_kwh", "Cumulative RAM energy (kWh)")
GPU_E = Gauge("training_gpu_energy_kwh", "Cumulative GPU energy from codecarbon (kWh)")
 
PHASES = ("dataset", "load_model", "fine_tuning")
for _p in PHASES:
    PHASE.labels(phase=_p).set(0)
 
POWER_CSV = os.path.join(CARBON_DIR, "power_timeseries.csv")
with open(POWER_CSV, "w") as _f:
    _f.write("timestamp,cpu_w,ram_w,cpu_e,ram_e,gpu_e,gpus,phase\n")
PHASE_ENERGY_CSV = os.path.join(CARBON_DIR, "phase_energy.csv")
with open(PHASE_ENERGY_CSV, "w") as _f:
    _f.write("timestamp,phase,energy\n")
 
_current_phase = {"name": "idle"}
 
 
class PromAndCsvLoggerOutput(LoggerOutput):
    """CodeCarbon LoggerOutput that also updates Prometheus gauges and CSV."""
 
    def _publish(self, total, delta):
        cpu_w  = float(getattr(delta, "cpu_power", 0.0) or 0.0)
        ram_w  = float(getattr(delta, "ram_power", 0.0) or 0.0)
        cpu_e  = float(getattr(total, "cpu_energy", 0.0) or 0.0)
        ram_e  = float(getattr(total, "ram_energy", 0.0) or 0.0)
        gpu_e  = float(getattr(total, "gpu_energy", 0.0) or 0.0)
        CPU_W.set(cpu_w)
        RAM_W.set(ram_w)
        CPU_E.set(cpu_e)
        RAM_E.set(ram_e)
        GPU_E.set(gpu_e)
        gpus_kw = 0.0
        GPU_W.set(gpus_kw)
        ts = datetime.now(timezone.utc).isoformat()
        with open(POWER_CSV, "a") as fh:
            fh.write(f"{ts},{cpu_w},{ram_w},{cpu_e},{ram_e},{gpu_e},{gpus_kw},{_current_phase['name']}\n")
 
    def out(self, total, delta):
        super().out(total, delta)
        self._publish(total, delta)
 
    def live_out(self, total, delta):
        try:
            super().live_out(total, delta)
        except AttributeError:
            pass
        self._publish(total, delta)
 
 
my_logger = PromAndCsvLoggerOutput(_logger, logging.INFO)
 
tracker = EmissionsTracker(
    project_name="bio-mistral-7b",
    output_dir=CARBON_DIR,
    save_to_file=True,
    on_csv_write="append",
    output_file="emissions.csv",
    tracking_mode="process",
    measure_power_secs=1,
    save_to_logger=True,
    logging_logger=my_logger,
)
tracker.start()
 
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
 
 
def begin_phase(name: str):
    _current_phase["name"] = name
    PHASE.labels(phase=name).set(1)
    monitor.begin_window(name)
 
 
def end_phase(name: str):
    energy = monitor.end_window(name)
    PHASE.labels(phase=name).set(0)
    _current_phase["name"] = "idle"
    ts = datetime.now(timezone.utc).isoformat()
    _logger.info(f"phase={name} energy={energy}")
    with open(PHASE_ENERGY_CSV, "a") as _f:
        _f.write(f"{ts},{name},{energy}\n")
    return energy
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 6. DATASET
# ──────────────────────────────────────────────────────────────────────────────
 
print("Loading and mapping dataset...")
begin_phase("dataset")
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(format_sample)
ds_energy = end_phase("dataset")
 
# ──────────────────────────────────────────────────────────────────────────────
# 7. LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
 
begin_phase("load_model")
 
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id
 
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
 
print(f"Downloading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
 
# Train / test split
dataset_splits = dataset.train_test_split(test_size=TEST_SIZE, seed=42)
train_dataset = dataset_splits["train"]
test_dataset  = dataset_splits["test"]
 
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"],
)
 
model = prepare_model_for_kbit_training(model)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
 
training_arguments = TrainingArguments(
    output_dir=RESULTS_DIR,
    report_to="wandb" if wandb_key else "none",
    eval_strategy="epoch",
    optim="paged_adamw_8bit",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    log_level="info",
    save_steps=500,
    logging_steps=20,
    learning_rate=2e-5,
    num_train_epochs=1,
    warmup_steps=100,
    lr_scheduler_type="constant",
    push_to_hub=PUSH_TO_HUB,
    hub_token=hf_token if PUSH_TO_HUB else None,
    hub_model_id=HUB_MODEL_ID if PUSH_TO_HUB else None,
    hub_strategy="end",
)
 
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
)
lmodel_energy = end_phase("load_model")
 
# ──────────────────────────────────────────────────────────────────────────────
# 8. TRAIN
# ──────────────────────────────────────────────────────────────────────────────
 
begin_phase("fine_tuning")
trainer.train()
 
print_trainable_parameters(model)
 
ft_energy = end_phase("fine_tuning")
emissions = tracker.stop()
 
# ──────────────────────────────────────────────────────────────────────────────
# 9. SAVE & PUSH
# ──────────────────────────────────────────────────────────────────────────────
 
# Save locally (persistent on Lightning AI)
trainer.model.save_pretrained(SAVED_MODEL_DIR)
tokenizer.save_pretrained(SAVED_MODEL_DIR)
print(f"Model saved to {SAVED_MODEL_DIR}")
 
# Push to Hub
if PUSH_TO_HUB and hf_token:
    trainer.model.push_to_hub(HUB_MODEL_ID, token=hf_token)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=hf_token)
    print(f"Model pushed to hub: {HUB_MODEL_ID}")
 
# ──────────────────────────────────────────────────────────────────────────────
# 10. PLOTS & ENERGY SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
 
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Lightning AI
import matplotlib.pyplot as plt
 
log_history = trainer.state.log_history
logs_df = pd.DataFrame(log_history)
 
train_loss = logs_df.dropna(subset=["loss"])
eval_loss  = logs_df.dropna(subset=["eval_loss"])
 
# --- Loss plot ---
plt.figure(figsize=(15, 8))
plt.plot(train_loss["epoch"], train_loss["loss"], label="Training Loss")
plt.plot(eval_loss["epoch"], eval_loss["eval_loss"], label="Validation Loss", marker="x")
plt.title("Loss Function — BioMistral-7B Fine-tuning")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_BASE, "loss_function.png")
plt.savefig(loss_plot_path, transparent=True, dpi=150)
plt.close()
print(f"Loss plot saved to {loss_plot_path}")
 
# --- Power consumption plot ---
try:
    power_df = pd.read_csv(POWER_CSV, parse_dates=["timestamp"])
    if not power_df.empty:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(power_df["timestamp"], power_df["cpu_w"], label="CPU (W)")
        ax.plot(power_df["timestamp"], power_df["ram_w"], label="RAM (W)")
 
        phase_colors = {
            "dataset": "tab:orange",
            "load_model": "tab:green",
            "fine_tuning": "tab:red",
        }
        for phase_name, color in phase_colors.items():
            mask = power_df["phase"] == phase_name
            if mask.any():
                ax.axvspan(
                    power_df.loc[mask, "timestamp"].min(),
                    power_df.loc[mask, "timestamp"].max(),
                    alpha=0.1,
                    color=color,
                    label=f"phase: {phase_name}",
                )
 
        ax.set_title("Power Consumption (CPU & RAM)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right")
        ax.grid(True)
        fig.tight_layout()
        power_plot_path = os.path.join(OUTPUT_BASE, "power_consumption.png")
        fig.savefig(power_plot_path, transparent=True, dpi=150)
        plt.close(fig)
        print(f"Power plot saved to {power_plot_path}")
except FileNotFoundError:
    print(f"No power time series found at {POWER_CSV}")
 
# --- Energy summary ---
print(f"\nEnergy loading dataset  : {ds_energy}")
print(f"Energy loading model    : {lmodel_energy}")
print(f"Energy in fine-tuning   : {ft_energy}")
 
energy_summary = pd.DataFrame([
    {"phase": "dataset",     "energy": str(ds_energy)},
    {"phase": "load_model",  "energy": str(lmodel_energy)},
    {"phase": "fine_tuning", "energy": str(ft_energy)},
])
energy_csv_path = os.path.join(OUTPUT_BASE, "energy_summary.csv")
energy_summary.to_csv(energy_csv_path, index=False)
print(f"Energy summary saved to {energy_csv_path}")
 
print(f"\n✅ BioMistral-7B fine-tuning complete!")
print(f"All outputs saved under: {OUTPUT_BASE}")