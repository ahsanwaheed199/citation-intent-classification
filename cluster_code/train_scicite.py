import json, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "data_lora/soft_scicite"
MODEL    = "Qwen/Qwen2.5-1.5B"
OUT      = "lora_soft_scicite"

label2id = json.load(open(f"{DATA_DIR}/label2id.json"))
id2label = {v: k for k, v in label2id.items()}

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

ds = load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl",
                                       "test":  f"{DATA_DIR}/test.jsonl"})
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)
ds = ds.remove_columns("text").rename_column("label", "labels")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id,
    dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id

lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                  lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=8,                       # upper limit; early stopping ends it sooner
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",                     # must save each epoch to keep the best
    load_best_model_at_end=True,               # roll back to the best epoch
    metric_for_best_model="eval_macro_f1",     # judge "best" by macro-F1 (imbalanced data)
    greater_is_better=True,
    save_total_limit=1,                        # don't fill the disk with checkpoints
    logging_steps=20,
    bf16=True,
    report_to="none")

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    processing_class=tok, data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])  # stop if no F1 gain for 2 epochs

trainer.train()
final = trainer.evaluate()
print("FINAL:", final)

with open(f"{OUT}_result.json", "w") as f:
    json.dump({"model": MODEL, "data": DATA_DIR,
               "accuracy": final["eval_accuracy"],
               "macro_f1": final["eval_macro_f1"]}, f, indent=2)
print("saved", f"{OUT}_result.json")
