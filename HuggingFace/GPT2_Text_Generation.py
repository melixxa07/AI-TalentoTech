# Steps:

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, device_map='auto')
## "auto" → Automatically selects the available device (GPU or CPU) for the model

# GPT-2 does not have a padding token, so set it manually
tokenizer.pad_token = tokenizer.eos_token


# -- TEXT GENERATION --
# 2. Generate text using the pre-trained model
def generate_text(prompt, max_lenght=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs.to(model.device), 
                            max_length=max_lenght, 
                            num_return_sequences=1,
                            no_repeat_ngram_size=2, 
                            top_k=50, top_p=0.95,
                            do_sample=True) 
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# def generate_text(prompt, max_length=100):
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#     outputs = model.generate(inputs.input_ids.to(model.device),
#                              attention_mask=inputs.attention_mask.to(model.device),
#                              max_length=max_length,
#                              num_return_sequences=1,
#                              no_repeat_ngram_size=2,
#                              top_k=50,
#                              top_p=0.95,
#                              do_sample=True,
#                              pad_token_id=tokenizer.eos_token_id)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 3. Generate initial text using the prompt "La energía solar es"
print(generate_text("La energía solar es"))


# 4. Prepare data for fine-tuning the model 
# (Note: Fine-tuning GPT-2 for text generation is typically done on large text datasets,
# but this example demonstrates the basic steps for fine-tuning)
train_data = [
    "La energía solar es una fuente renovable que aprovecha la radiación del sol.",
    "Los aerogeneradores convierten la energía eólica en electricidad limpia.",
    "La biomasa utiliza materia orgánica para producir energía renovable.",
    "La energía geotérmica aprovecha el calor del interior de la Tierra.",
    "Las centrales hidroeléctricas generan energía a partir del flujo de agua.",
]

# Save the training data to a text file
train_path = "HuggingFace/train.txt"
with open(train_path, "w", encoding="utf-8-sig") as file:
    for line in train_data:
        file.write(line + "\n")
## This code writes the contents of train_data to a file named "train.txt"
## Opens the file "train.txt" in write mode ("w"), which creates the file if it doesn’t exist or overwrites it if it does.
## Writes each string to the file, adding a newline ("\n") after each entry to ensure that each text appears on a separate line.       
## encoding="utf-8-sig" is used to specify the character encoding of the file. In this case it shuold read chacarcters like á, é, í, ó, ú, ñ, etc.
        
# -- DATA PREPROCESSING --
# 5. Load the dataset 
dataset = load_dataset("text", data_files={"train": train_path})["train"]
 
# Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    #tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Labels are same as input_ids for causal LM
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)


#6. Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  
## data_collator is used in training to batch-process the dataset efficiently


# 7. Training arguments
training_args = TrainingArguments(
    output_dir='HuggingFace/gpt2-energy',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)
    
    
# 8. Create the trainer
trainer = Trainer(
    model= model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 9. Fine-tune/train the model
trainer.train()

# 10. Generate text using the fine-tuned model
print(generate_text("La energía renovable es importante porque"))
## This code generates text using the prompt "La energía nuclear es" with the fine-tuned model.

