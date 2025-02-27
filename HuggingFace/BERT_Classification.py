# Steps:

# 1. Install the transformers library from Hugging Face and load a pre-trained BERT model.
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# 1.1. Prepare a sample dataset
texts = [
    "Los paneles solares son una fuente eficiente de energía renovable.",
    "El carbón sigue siendo una fuente importante de energía en muchos países.",
    "La energía eólica está ganando popularidad en todo el mundo.",
    "Las centrales nucleares son controversiales pero producen energía sin emisiones de CO2.",
    "La biomasa es una forma de energía renovable que utiliza materiales orgánicos.",
]

labels = [1, 0, 1, 0, 1]  # 1: Renewable, 0: Non-renewable

# 1.2. Create a Hugging Face dataset
dataset = Dataset.from_dict({"text": texts, "label": labels})


# 2. Load a pre-trained BERT model from Hugging Face.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
## "Base" → 12 layers, 110 million parameters (lighter than BERT-large) 
## "Uncased" → All text is lowercased (e.g., "Apple" → "apple")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
## num_labels=2: Specifies that the model is being fine-tuned for binary classification (e.g., positive vs. negative sentiment)


# --- Check for GPU availability ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- Move the model to the available device (GPU or CPU) ---
model.to(device)


# 3. Function for PREPROCESSING the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)


# 4. Split the dataset into training and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)


# 5. Function for computing metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="HuggingFace/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    overwrite_output_dir=True,
)

# 7. Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 8. Train the model
trainer.train()

# 9. Evaluate the model
eval_results = trainer.evaluate()
#print(eval_results)

# 10. Make predictions
#text = 'La energía geotérmica aprovecha el calor de la Tierra.'
#text = 'La energía obtenida a través del hidrógeno es una de las formas más innovadoras de los últimos años.'
text = 'El petróleo es una fuente de energía muy usada en el mundo.'
inputs = tokenizer(text, return_tensors="pt")

# Let's do this to make sure all tensors are on the same device
# This can happen when we use the GPU. If the model is on the GPU, we need to move the inputs to the GPU as well
## Hugging Face's Trainer or DataLoader may load tensors on the CPU by default, while the model runs on the GPU. Since PyTorch requires all tensors to be on the same device, this mismatch can cause errors.
# --- Move inputs to the same device as the model ---
inputs = {key: value.to(device) for key, value in inputs.items()}
# --- Make prediction ---
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits).item()
print(f'Predicción: {"✅ Renewable Energy" if prediction == 1 else "❌ Non-renewable Energy"}')