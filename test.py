from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from optimum.quanto import QuantizedModelForCausalLM, qint4
import os
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load env
load_dotenv()
access_token = os.getenv('HF_TOKEN')

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
model = QuantizedModelForCausalLM.quantize(model, weights=qint4)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side="right")
tokenizer.pad_token = "<|finetune_right_pad_id|>"


# model.gradient_checkpointng_enable() # reduce vram by not storing intermediate values

# Apply LoRA
lora_config = LoraConfig(task_type="CAUSAL_LM", 
                         r=8, lora_alpha=32, 
                         lora_dropout=0.1, 
                         target_modules=["q_proj", "v_proj"],
                         bias="none")
model = get_peft_model(model, lora_config)

# Clean dataset
def format_dataset(data_point):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Based on input title generate the prompt for an AI model to simulate the role of the input title
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {data_point['act']}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {data_point['prompt']}
    <|eot_id|>
"""
    tokens = tokenizer(prompt,
        truncation=True,
        max_length=256,
        padding="max_length")
    return tokens

# Load dataset
dataset = load_dataset("fka/awesome-chatgpt-prompts", split='train')
dataset = dataset.map(format_dataset)
dataset = dataset.remove_columns(["act", "prompt"])

# Split into train and test set
dataset = dataset.train_test_split(test_size=0.1)

# batcher
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer setup
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-5, 
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_8bit",
    warmup_steps=2,
    remove_unused_columns=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)

# Train
trainer.train()