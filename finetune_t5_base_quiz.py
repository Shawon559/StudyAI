import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

# =========================
# CONFIG
# =========================
MODEL_NAME = "t5-base"              # base model to fine-tune
CSV_PATH = "quiz_dataset.csv"       # your CSV in this folder
OUTPUT_DIR = "t5-base-quiz-finetuned"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# =========================
# 1) LOAD DATA
# =========================
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Expect columns: 'input', 'target'
if not {"input", "target"}.issubset(df.columns):
    raise ValueError(f"CSV must contain columns 'input' and 'target'. Found: {df.columns.tolist()}")

df = df.dropna(subset=["input", "target"])
print("Dataset rows:", len(df))

dataset = Dataset.from_pandas(df[["input", "target"]])

# Split into train / validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]

print("Train size:", len(train_ds))
print("Val size:", len(val_ds))

# =========================
# 2) LOAD TOKENIZER + MODEL
# =========================
print("Loading tokenizer and model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Optional: add a task prefix so the model knows what we want
TASK_PREFIX = "generate quiz questions: "


def preprocess_function(batch):
    # Add prefix to input
    inputs = [TASK_PREFIX + x for x in batch["input"]]

    # Encode inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    # Encode targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing train set...")
train_tokenized = train_ds.map(
    preprocess_function,
    batched=True,
    remove_columns=["input", "target"],
)

print("Tokenizing val set...")
val_tokenized = val_ds.map(
    preprocess_function,
    batched=True,
    remove_columns=["input", "target"],
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# =========================
# 3) TRAINING ARGUMENTS
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # basic train/eval config
    do_train=True,
    do_eval=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,

    # optimization
    learning_rate=5e-5,
    weight_decay=0.01,

    # logging & saving (older-style)
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,

    # keep it CPU/MPS-friendly
    fp16=False,
)

# =========================
# 4) TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving model to:", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# =========================
# 5) QUICK TEST GENERATION
# =========================
print("\nTesting generation on one example...\n")

sample_input = train_ds[0]["input"]
print("INPUT (truncated):", sample_input[:200], "...\n")

encoded = tokenizer(
    TASK_PREFIX + sample_input,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_INPUT_LENGTH,
)

output_ids = model.generate(
    **encoded,
    max_length=MAX_TARGET_LENGTH,
    num_beams=4,
    early_stopping=True,
)

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("GENERATED QUIZ TEXT:\n", generated)
print("\nDone.")