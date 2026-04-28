import sys
import os
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import torch

pkgs = ['python-dotenv', 'trackio', 'transformers>=5.0', 'trl[peft]>=1.0.0', 'accelerate', 'bitsandbytes', 'datasets', 'ninja', 'codecarbon', 'packaging', 'zeus', 'wandb', 'huggingface-hub', 'tqdm', 'evaluate', 'bert_score']

def install_packages(packages):
    print("Resolving environment and installing packages via uv...")
    try:
        subprocess.check_call(['uv', 'pip', 'install'] + packages)
        print("All packages successfully installed and verified!")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")

install_packages(pkgs)

from dotenv import load_dotenv
from huggingface_hub import login, auth_list
import wandb

load_dotenv()
wandb_key = os.getenv('WANDB')
hf_token = os.getenv('HF_TOKEN')

wandb.login(key=wandb_key)
login(token=hf_token, add_to_git_credential=False)

auth_list()

import json
from datetime import datetime, timezone
from datasets import load_dataset
import urllib.request
import threading
import time
import re
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from accelerate import Accelerator
from trl import SFTTrainer, setup_chat_format
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging
from zeus.device import get_gpus
from zeus.monitor import ZeusMonitor
from prometheus_client import start_http_server, Gauge

gpus = get_gpus()
print(gpus)

accelerator = Accelerator()


def format_chat_template(row):
    row_json = [{"role": "user", "content": row["instruction"] + " " + row["input"]},
                {"role": "assistant", "content": row["output"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

if is_main:
    os.makedirs('code_carbon_bio-llama3-8b-int8', exist_ok=True)

    log_name = "bio-llama3-8b-int8-logs"
    _logger = logging.getLogger(log_name)
    _channel = logging.FileHandler(log_name + '.log')
    _logger.addHandler(_channel)
    _logger.setLevel(logging.INFO)

    start_http_server(8000)

    PHASE = Gauge('training_phase', 'Current fine-tuning phase (1=active, 0=inactive)', ['phase'])
    CPU_W = Gauge('training_cpu_power_watts', 'CPU power from codecarbon (watts)')
    RAM_W = Gauge('training_ram_power_watts', 'RAM power from codecarbon, estimate (watts)')
    GPU_W = Gauge('training_gpu_power_watts', 'GPU power total from amd_dme (watts)')
    CPU_E = Gauge('training_cpu_energy_kwh', 'Cumulative CPU energy (kWh)')
    RAM_E = Gauge('training_ram_energy_kwh', 'Cumulative RAM energy (kWh)')
    GPU_E = Gauge('training_gpu_energy_kwh', 'Cumulative GPU energy from codecarbon (kWh)')

    PHASES = ('dataset', 'load_model', 'fine_tuning', 'evaluation')
    for _p in PHASES:
        PHASE.labels(phase=_p).set(0)

    POWER_CSV = './code_carbon_bio-llama3-8b-int8/power_timeseries.csv'
    with open(POWER_CSV, 'w') as _f:
        _f.write('timestamp,cpu_w,ram_w,gpu_w,phase\n')

    _current_phase = {'name': 'idle'}

    AMD_DME_URL = os.environ.get('AMD_DME_URL', 'http://amd_dme:5000/metrics')
    _gpu_power_re = re.compile(r'^gpu_power_usage(?:_watts)?\{[^}]*\}\s+([0-9.eE+-]+)', re.MULTILINE)

    def _read_amd_gpu_power_w():
        try:
            with urllib.request.urlopen(AMD_DME_URL, timeout=2) as r:
                text = r.read().decode('utf-8', errors='ignore')
            vals = [float(m.group(1)) for m in _gpu_power_re.finditer(text)]
            return sum(vals) if vals else 0.0
        except Exception:
            return 0.0

    def _gpu_power_loop(stop_evt):
        while not stop_evt.is_set():
            w = _read_amd_gpu_power_w()
            GPU_W.set(w)
            stop_evt.wait(1.0)

    _gpu_stop = threading.Event()
    _gpu_thread = threading.Thread(target=_gpu_power_loop, args=(_gpu_stop,), daemon=True)
    _gpu_thread.start()


    class PromAndCsvLoggerOutput(LoggerOutput):
        """codecarbon LoggerOutput that ALSO updates Prometheus gauges and appends to power_timeseries.csv on every flush."""

        def _publish(self, total, delta):
            cpu_w = float(getattr(delta, 'cpu_power', 0.0) or 0.0)
            ram_w = float(getattr(delta, 'ram_power', 0.0) or 0.0)
            CPU_W.set(cpu_w)
            RAM_W.set(ram_w)
            CPU_E.set(float(getattr(total, 'cpu_energy', 0.0) or 0.0))
            RAM_E.set(float(getattr(total, 'ram_energy', 0.0) or 0.0))
            GPU_E.set(float(getattr(total, 'gpu_energy', 0.0) or 0.0))
            gpu_w = _read_amd_gpu_power_w()
            ts = datetime.now(timezone.utc).isoformat()
            with open(POWER_CSV, 'a') as fh:
                fh.write(f"{ts},{cpu_w},{ram_w},{gpu_w},{_current_phase['name']}\n")

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
        project_name='bio-llama3-8b-int8',
        output_dir="./code_carbon_bio-llama3-8b-int8/",
        save_to_file=True,
        on_csv_write='append',
        output_file="emissions.csv",
        tracking_mode="process",
        measure_power_secs=1,
        save_to_logger=True,
        logging_logger=my_logger
    )
    tracker.start()

monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])


def begin_phase(name: str):
    """Mark a fine-tuning phase active: set Prometheus gauge to 1 and start a Zeus window on every rank."""
    monitor.begin_window(name)
    if is_main:
        _current_phase['name'] = name
        PHASE.labels(phase=name).set(1)


def end_phase(name: str):
    """End a fine-tuning phase: close the Zeus window on every rank and clear the Prometheus gauge."""
    energy = monitor.end_window(name)
    if is_main:
        PHASE.labels(phase=name).set(0)
        _current_phase['name'] = 'idle'
    return energy


begin_phase('load_model')
model_name = "meta-llama/Llama-3.1-8B"
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

compute_dtype = torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

print('Downloading model in 8-bit (LLM.int8)')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": accelerator.local_process_index},
    attn_implementation="eager",
)

model, tokenizer = setup_chat_format(model, tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lmodel_energy = end_phase('load_model')

print('Mapping dataset')
begin_phase('dataset')
dataset = load_dataset("bio-nlp-umass/bioinstruct", split="train")
dataset = dataset.map(format_chat_template)
dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset = dataset_splits['train']
test_dataset = dataset_splits['test']
ds_energy = end_phase('dataset')

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

training_arguments = TrainingArguments(
    output_dir="./results-int8",
    report_to="wandb",
    eval_strategy="epoch",
    num_train_epochs=1,
    optim="adamw_torch_fused",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    log_level="info",
    save_steps=500,
    logging_steps=20,
    learning_rate=1e-4,
    warmup_steps=10,
    lr_scheduler_type="constant",
    bf16=True,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    args=training_arguments,
    peft_config=peft_config,
)

begin_phase('fine_tuning')
trainer.train()

print_trainable_parameters(model)
ft_energy = end_phase('fine_tuning')

if is_main:
    new_model = 'Bio-llama3-8b-int8'
    trainer.model.save_pretrained(new_model)
    trainer.model.push_to_hub("darmasrmz/bio-llama3-8b-int8", token=hf_token)
    tokenizer.push_to_hub("darmasrmz/bio-llama3-8b-int8", token=hf_token)


accelerator.wait_for_everyone()

evaluation_energy = None
if is_main:
    begin_phase('evaluation')
    import evaluate
    from tqdm import tqdm

    bertscore = evaluate.load("bertscore")

    eval_dataset = load_dataset("bio-nlp-umass/bioinstruct", split="test")
    eval_splits = eval_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    eval_test = eval_splits['test']

    predictions = []
    references = []

    print("Starting generation...")
    model.config.use_cache = True
    model.eval()
    for row in tqdm(eval_test):
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{row['instruction']}

### Input:
{row['input']}

### Response:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        predictions.append(response)
        references.append(row['output'])

    print("Calculating metrics...")
    bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(f"BERTScore F1 (Mean): {sum(bertscore_results['f1']) / len(bertscore_results['f1'])}")

    evaluation_energy = end_phase('evaluation')
    emissions = tracker.stop()
    _gpu_stop.set()

    import pandas as pd
    import matplotlib.pyplot as plt

    log_history = trainer.state.log_history
    logs_df = pd.DataFrame(log_history)

    train_loss = logs_df.dropna(subset=['loss'])
    eval_loss = logs_df.dropna(subset=['eval_loss'])

    plt.figure(figsize=(15, 8))
    plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss')
    plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Validation Loss', marker='x')
    plt.title('Función de pérdida (int8)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_function_int8.png', transparent=True)
    plt.close()

    try:
        power_df = pd.read_csv(POWER_CSV, parse_dates=['timestamp'])
        if not power_df.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(power_df['timestamp'], power_df['cpu_w'], label='CPU (W)')
            ax.plot(power_df['timestamp'], power_df['ram_w'], label='RAM (W)')
            ax.plot(power_df['timestamp'], power_df['gpu_w'], label='GPU (W)')

            phase_colors = {'dataset': 'tab:orange', 'load_model': 'tab:green', 'fine_tuning': 'tab:red', 'evaluation': 'tab:blue'}
            for phase_name, color in phase_colors.items():
                mask = power_df['phase'] == phase_name
                if mask.any():
                    ax.axvspan(power_df.loc[mask, 'timestamp'].min(),
                               power_df.loc[mask, 'timestamp'].max(),
                               alpha=0.1, color=color, label=f'phase: {phase_name}')

            ax.set_title('Consumo de potencia (CPU, RAM, GPU) — int8')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Potencia (W)')
            ax.legend(loc='upper right')
            ax.grid(True)
            fig.tight_layout()
            fig.savefig('power_consumption_int8.png', transparent=True)
            plt.close(fig)
    except FileNotFoundError:
        print(f'No power time series found at {POWER_CSV}')

    print(f'Energy loading dataset: {ds_energy}')
    print(f'Energy loading model: {lmodel_energy}')
    print(f'Energy in fine-tuning model: {ft_energy}')
    print(f'Energy in evaluation: {evaluation_energy}')
