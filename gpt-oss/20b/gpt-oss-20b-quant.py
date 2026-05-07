import sys
import os
import subprocess

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION — edit these to match your experiment
# ──────────────────────────────────────────────────────────────────────────────

# Model -----------------------------------------------------------------------
MODEL_NAME           = "unsloth/gpt-oss-20b-bnb-4bit"  # 4-bit Unsloth version (~14 GB VRAM)
# Alternatives:
#   "unsloth/gpt-oss-20b-BF16"        — BF16 LoRA, needs ≥44 GB VRAM
#   "unsloth/gpt-oss-120b-bnb-4bit"   — 120B QLoRA, needs ≥65 GB VRAM
LOAD_IN_4BIT         = True   # Set False + use BF16 model if you have ≥44 GB VRAM
MAX_SEQ_LENGTH       = 2048  # Increase if you have VRAM headroom (2048, 4096, etc.)
DTYPE                = None   # None = auto-detect (bf16 on Ampere+, fp16 otherwise)
REASONING_EFFORT     = "medium"  # "low", "medium", "high" — controls gpt-oss reasoning depth

# LoRA ------------------------------------------------------------------------
LORA_R               = 16
LORA_ALPHA           = 16
LORA_DROPOUT         = 0.0   # Unsloth recommends 0 for optimized kernels
TARGET_MODULES       = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training ---------------------------------------------------------------------
NUM_TRAIN_EPOCHS     = 1
MAX_STEPS            = -1     # Set >0 to cap total steps (overrides epochs)
PER_DEVICE_BATCH     = 4
GRAD_ACCUM_STEPS     = 32
LEARNING_RATE        = 2e-5
LR_SCHEDULER         = "constant"
WARMUP_STEPS         = 100
LOGGING_STEPS        = 20
SAVE_STEPS           = 500
EVAL_STRATEGY        = "epoch"
REPORT_TO            = "wandb"

# Hub --------------------------------------------------------------------------
HUB_MODEL_ID         = "darmasrmz/bio-gpt-oss-20b-quant"  # Change to your repo
HUB_STRATEGY         = "end"
PUSH_TO_HUB          = True

# Dataset ----------------------------------------------------------------------
DATASET_NAME         = "bio-nlp-umass/bioinstruct"
TEST_SIZE            = 0.2

# Paths — Lightning AI persistent storage --------------------------------------
# Everything under /teamspace/studios/this_studio/ persists across restarts.
# /home/zeus/... is ephemeral and resets when the Studio restarts.
OUTPUT_BASE          = "/teamspace/studios/this_studio/gpt_oss_finetune"
RESULTS_DIR          = os.path.join(OUTPUT_BASE, "results")
CARBON_DIR           = os.path.join(OUTPUT_BASE, "code_carbon_gpt_oss")
SAVED_MODEL_DIR      = os.path.join(OUTPUT_BASE, "bio-gpt-oss-20b-quant")

# ──────────────────────────────────────────────────────────────────────────────
# 1. INSTALL PACKAGES
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# 1. INSTALL PACKAGES
# ──────────────────────────────────────────────────────────────────────────────

def install_packages():
    print("Installing Unsloth + deps from GitHub...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
        "transformers>=4.51.3,<=5.5.0",
        "triton>=3.4.0",
        "bitsandbytes",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "python-dotenv", "trackio", "accelerate", "datasets",
        "ninja", "codecarbon", "packaging", "zeus-ml", "wandb",
        "huggingface-hub", "tqdm", "evaluate", "nltk", "pandas",
        "matplotlib", "prometheus-client",
    ])
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "amdsmi"])
    print("All packages installed!")

install_packages()

# ──────────────────────────────────────────────────────────────────────────────
# 2. AUTH — Lightning AI injects secrets as env vars (Settings → Secrets)
# ──────────────────────────────────────────────────────────────────────────────
#
# In Lightning AI Studio:
#   1. Go to Settings (gear icon) → Secrets
#   2. Add two secrets:
#        Name: WANDB      Value: <your wandb api key>
#        Name: HF_TOKEN   Value: <your huggingface token>
#   3. They will be available as os.environ["WANDB"] / os.environ["HF_TOKEN"]
#
# If you're running locally instead, create a .env file with:
#   WANDB=wk_xxx
#   HF_TOKEN=hf_xxx

from dotenv import load_dotenv
load_dotenv()  # no-op on Lightning AI (no .env), but harmless

from huggingface_hub import login
import wandb

wandb_key = os.environ.get("WANDB")
hf_token  = os.environ.get("HF_TOKEN")

if not wandb_key:
    print("⚠️  WANDB secret not found. Disabling W&B logging.")
    REPORT_TO = "none"
else:
    wandb.login(key=wandb_key)

if not hf_token:
    print("⚠️  HF_TOKEN secret not found. Hub push will be disabled.")
    PUSH_TO_HUB = False
else:
    login(token=hf_token, add_to_git_credential=False)

# ──────────────────────────────────────────────────────────────────────────────
# 3. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import torch
from datetime import datetime, timezone
from datasets import load_dataset
import transformers.modeling_utils as _mu
_mu.PreTrainedModel._initialize_missing_keys = lambda self, *args, **kwargs: None
from unsloth import FastLanguageModel, standardize_sharegpt
from trl import SFTTrainer, SFTConfig
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
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

is_main = True

gpus = get_gpus()
print(f"Zeus GPUs    : {gpus}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def create_conversation(sample):
    """Convert BioInstruct instruction/input/output → ShareGPT messages format."""
    user_content = sample["instruction"]
    if sample.get("input") and len(sample["input"].strip()) > 0:
        user_content += "\n" + sample["input"]
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["output"]},
        ]
    }


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
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

if is_main:
    log_name = "bio_gpt_oss_20b_quant_logs"
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

    PHASES = ("dataset", "load_model", "fine_tuning", "evaluation")
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
        project_name="bio-gpt-oss-20b-quant",
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
    monitor.begin_window(name)
    if is_main:
        _current_phase["name"] = name
        PHASE.labels(phase=name).set(1)


def end_phase(name: str):
    energy = monitor.end_window(name)
    if is_main:
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
dataset = dataset.map(create_conversation, remove_columns=[], batched=False)

# Unsloth's standardize_sharegpt normalizes the messages column
dataset = standardize_sharegpt(dataset)

ds_energy = end_phase("dataset")

# ──────────────────────────────────────────────────────────────────────────────
# 7. LOAD MODEL (Unsloth FastLanguageModel)
# ──────────────────────────────────────────────────────────────────────────────

print(f"Loading model: {MODEL_NAME}")
begin_phase("load_model")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# Apply LoRA adapters via Unsloth (automatically targets the right layers)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print_trainable_parameters(model)

# Format dataset using gpt-oss Harmony chat template
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort=REASONING_EFFORT,
        )
        for convo in convos
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)

# Train / test split
dataset_splits = dataset.train_test_split(test_size=TEST_SIZE, shuffle=True, seed=42)
train_dataset = dataset_splits["train"]
test_dataset  = dataset_splits["test"]

lmodel_energy = end_phase("load_model")

print(f"Train samples : {len(train_dataset)}")
print(f"Test samples  : {len(test_dataset)}")
print("First formatted example (truncated):")
print(train_dataset[0]["text"][:500], "…\n")

# ──────────────────────────────────────────────────────────────────────────────
# 8. TRAINER
# ──────────────────────────────────────────────────────────────────────────────

training_args = SFTConfig(
    output_dir=RESULTS_DIR,
    report_to=REPORT_TO,
    eval_strategy=EVAL_STRATEGY,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS,
    optim="adamw_8bit",
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    bf16=True,
    log_level="info",
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type=LR_SCHEDULER,
    weight_decay=0.01,
    seed=3407,
    push_to_hub=PUSH_TO_HUB,
    hub_token=hf_token if PUSH_TO_HUB else None,
    hub_model_id=HUB_MODEL_ID if PUSH_TO_HUB else None,
    hub_strategy=HUB_STRATEGY,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
)

# ──────────────────────────────────────────────────────────────────────────────
# 9. TRAIN
# ──────────────────────────────────────────────────────────────────────────────

begin_phase("fine_tuning")
trainer.train()
print_trainable_parameters(model)
ft_energy = end_phase("fine_tuning")

# ──────────────────────────────────────────────────────────────────────────────
# 10. SAVE & PUSH
# ──────────────────────────────────────────────────────────────────────────────

if is_main:
    tracker.stop()

    # Save locally (persistent on Lightning AI)
    model.save_pretrained(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)
    print(f"Model saved to {SAVED_MODEL_DIR}")

    # Push to Hub
    if PUSH_TO_HUB and hf_token:
        model.push_to_hub(HUB_MODEL_ID, token=hf_token)
        tokenizer.push_to_hub(HUB_MODEL_ID, token=hf_token)
        print(f"Model pushed to hub: {HUB_MODEL_ID}")

# ──────────────────────────────────────────────────────────────────────────────
# 11. PLOTS & ENERGY SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

if is_main:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for Lightning AI
    import matplotlib.pyplot as plt

    log_history = trainer.state.log_history
    logs_df = pd.DataFrame(log_history)

    train_loss = logs_df.dropna(subset=["loss"])
    eval_loss  = logs_df.dropna(subset=["eval_loss"])

    plt.figure(figsize=(15, 8))
    plt.plot(train_loss["epoch"], train_loss["loss"], label="Training Loss")
    plt.plot(eval_loss["epoch"], eval_loss["eval_loss"], label="Validation Loss", marker="x")
    plt.title("Loss Function — gpt-oss-20b Fine-tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(OUTPUT_BASE, "loss_function.png")
    plt.savefig(loss_plot_path, transparent=True, dpi=150)
    print(f"Loss plot saved to {loss_plot_path}")

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

    print("\n✅ Fine-tuning complete!")
    print(f"All outputs saved under: {OUTPUT_BASE}")