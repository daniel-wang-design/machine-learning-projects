from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import os
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load env
load_dotenv()
access_token = os.getenv('HF_TOKEN')

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, lora_config)

# Clean dataset
def format_dataset(data_point):
    prompt = f"""###SYSTEM: Based on INPUT title generate the prompt for generative model

###INPUT: {data_point['act']}

###PROMPT: {data_point['prompt']}
"""
    tokens = tokenizer(prompt,
        truncation=True,
        max_length=256,
        padding="max_length",)
    tokens["labels"] = tokens['input_ids'].copy()
    return tokens

# Load dataset
dataset = load_dataset("fka/awesome-chatgpt-prompts", split='train')
dataset = dataset.map(format_dataset)
dataset = dataset.remove_columns(['act', "prompt"])

# Split into train and test set
dataset = dataset.train_test_split(test_size=0.1)

# Trainer setup
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    use_cpu=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train
trainer.train()