import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import subprocess
import torch

pkgs = ['python-dotenv', 'openai-harmony', 'trackio', 'triton>=3.4.0', 'transformers>=5.0.0', 'trl>=1.0.0', 'accelerate', 'unsloth_zoo', 'unsloth', 'bitsandbytes', 'datasets', 'ninja', 'peft', 'codecarbon', 'packaging', 'zeus', 'wandb', 'huggingface-hub', 'tqdm', 'evaluate', 'bert_score', 'prometheus-client']

def install_packages(packages):
    try:
        subprocess.check_call(['uv', 'pip', 'install'] + packages)
        print('All packages installed successfully')
    except subprocess.CalledProcessError as e:
        print(f'Error installing packages: {e}')

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

from openai_harmony import (
    load_harmony_encoding, HarmonyEncodingName,
    Conversation, Role, Message,
    SystemContent, DeveloperContent, ReasoningEffort
)
import json
from datetime import datetime
from datasets import load_dataset
from unsloth import FastLanguageModel

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging
from zeus.device import get_gpus
from zeus.monitor import ZeusMonitor
from prometheus_client import start_http_server, Gauge
from unsloth.chat_templates import standardize_sharegpt

gpus = get_gpus()
print(gpus)
max_seq_length = 1024
dtype = None

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

os.makedirs('code_carbon_gpt_oss_20b', exist_ok=True)

log_name = "bio_gpt_oss_20b_logs"
_logger = logging.getLogger(log_name)
_channel = logging.FileHandler(log_name + '.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

start_http_server(8000)

PHASE = Gauge('training_phase', 'Current fine-tuning phase (1=active, 0=inactive)', ['phase'])
CPU_W = Gauge('training_cpu_power_watts', 'CPU power from codecarbon (watts)')
RAM_W = Gauge('training_ram_power_watts', 'RAM power from codecarbon, estimate (watts)')
CPU_E = Gauge('training_cpu_energy_kwh', 'Cumulative CPU energy (kWh)')
RAM_E = Gauge('training_ram_energy_kwh', 'Cumulative RAM energy (kWh)')

PHASES = ('dataset', 'load_model', 'fine_tuning')
for _p in PHASES:
    PHASE.labels(phase=_p).set(0)

POWER_CSV = './code_carbon_gpt_oss_20b/power_timeseries.csv'
with open(POWER_CSV, 'w') as _f:
    _f.write('timestamp,cpu_w,ram_w,phase\n')

_current_phase = {'name': 'idle'}

class PromAndCsvLoggerOutput(LoggerOutput):
    """codecarbon LoggerOutput that ALSO updates Prometheus gauges and appends to power_timeseries.csv on every flush."""

    def _publish(self, total, delta):
        cpu_w = float(getattr(delta, 'cpu_power', 0.0) or 0.0)
        ram_w = float(getattr(delta, 'ram_power', 0.0) or 0.0)
        CPU_W.set(cpu_w)
        RAM_W.set(ram_w)
        CPU_E.set(float(getattr(total, 'cpu_energy', 0.0) or 0.0))
        RAM_E.set(float(getattr(total, 'ram_energy', 0.0) or 0.0))
        try:
            with open(POWER_CSV, 'a') as fh:
                fh.write(f"{datetime.utcnow().isoformat()},{cpu_w},{ram_w},{_current_phase['name']}\n")
        except Exception:
            pass

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
    project_name = 'bio-gpt-oss-20b',
    output_dir="./code_carbon_gpt_oss_20b/",
    save_to_file=True,
    on_csv_write='append',
    output_file="emissions.csv",
    tracking_mode="process",
    measure_power_secs=1,
    save_to_logger=True,
    logging_logger=my_logger
)

def begin_phase(name: str):
    """Mark a fine-tuning phase active: set Prometheus gauge to 1 and start a Zeus window."""
    _current_phase['name'] = name
    PHASE.labels(phase=name).set(1)
    monitor.begin_window(name)


def end_phase(name: str):
    """End a fine-tuning phase: close the Zeus window and clear the Prometheus gauge."""
    energy = monitor.end_window(name)
    PHASE.labels(phase=name).set(0)
    _current_phase['name'] = 'idle'
    return energy

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


tracker.start()
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

def render_pair_harmony(question, answer):
    convo = Conversation.from_messages([
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(
                "You are a biomedical expert with advanced knowledge in clinical reasoning and diagnostics. "
                "Respond with ONLY the final diagnosis/cause in ≤15 words."
            )
        ),
        Message.from_role_and_content(Role.USER, question.strip()),
        Message.from_role_and_content(Role.ASSISTANT, answer.strip()),
    ])
    tokens = enc.render_conversation(convo)
    text = enc.decode(tokens)
    return text

def prompt_style_harmony(examples):
    qs = examples["instruction"]
    inputs = examples["input"]
    ans = examples["output"]
    outputs = {"text": []}
    for q, i, a in zip(qs, inputs, ans):
        rendered = render_pair_harmony(q+i, a)
        outputs["text"].append(rendered)
    return outputs

def formatting_prompts_func(examples):
   convos = examples["text"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }

print('Mapping dataset')
begin_phase('dataset')
dataset = load_dataset("bio-nlp-umass/bioinstruct", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(prompt_style_harmony, batched=True)
dataset = dataset.map(formatting_prompts_func, batched=True)

ds_energy = end_phase('dataset')

begin_phase('load_model')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = dtype,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    full_finetuning = False,
    token = hf_token,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

lmodel_energy = end_phase('load_model')

begin_phase('fine_tuning')

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        max_steps = None,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",
    ),
)

from unsloth.chat_templates import train_on_responses_only

gpt_oss_kwargs = dict(instruction_part = "<|start|>user<|message|>", response_part = "<|start|>assistant<|channel|>final<|message|>")

trainer = train_on_responses_only(
    trainer,
    **gpt_oss_kwargs,
)
trainer_stats = trainer.train()
print_trainable_parameters(model)

ft_energy = end_phase('fine_tuning')
emissions = tracker.stop()

import pandas as pd
import matplotlib.pyplot as plt

log_history = trainer.state.log_history
logs_df = pd.DataFrame(log_history)

train_loss = logs_df.dropna(subset=['loss'])
eval_loss = logs_df.dropna(subset=['eval_loss'])

plt.figure(figsize=(15, 8))
plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss')
plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Validation Loss', marker='x')
plt.title('Función de pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()
plt.tight_layout()
plt.savefig('loss_function.png', transparent=True)

try:
    power_df = pd.read_csv(POWER_CSV, parse_dates=['timestamp'])
    if not power_df.empty:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(power_df['timestamp'], power_df['cpu_w'], label='CPU (W)')
        ax.plot(power_df['timestamp'], power_df['ram_w'], label='RAM (W)')

        phase_colors = {'dataset': 'tab:orange', 'load_model': 'tab:green', 'fine_tuning': 'tab:red'}
        for phase_name, color in phase_colors.items():
            mask = power_df['phase'] == phase_name
            if mask.any():
                ax.axvspan(power_df.loc[mask, 'timestamp'].min(),
                           power_df.loc[mask, 'timestamp'].max(),
                           alpha=0.1, color=color, label=f'phase: {phase_name}')

        ax.set_title('Consumo de potencia del contenedor (CPU y RAM)')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Potencia (W)')
        ax.legend(loc='upper right')
        ax.grid(True)
        fig.tight_layout()
        fig.savefig('power_consumption.png', transparent=True)
        plt.close(fig)
except FileNotFoundError:
    print(f'No power time series found at {POWER_CSV}')

print(f'Energy loading dataset: {ds_energy}')
print(f'Energy loading model: {lmodel_energy}')
print(f'Energy in fine-tuning model: {ft_energy}')

model.push_to_hub("darmasrmz/bio-gpt-oss-20b-lora", token = hf_token)