import os
import math
import random
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dataset import maskedABSA_Dataset
from models import StanceBERT
from metrics import masked_bce_loss, masked_accuracy
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 8
LR = 3e-5
EPOCHS = 3
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 1
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    texts = [ex['text'] for ex in examples]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    word_one_hot = torch.stack([ex['word_one_hot_vector'] for ex in examples])  # (B, A)
    stance = torch.stack([ex['stance_vector'] for ex in examples])              # (B, A) with -1 for unknown

    # mask to compute loss only where: word present & label known (not -1)
    known_mask = (word_one_hot == 1.0) & (stance != -1.0)

    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])),
        "word_one_hot": word_one_hot,
        "stance": stance,
        "known_mask": known_mask
    }
    return batch



train_ds = maskedABSA_Dataset("Dataset/weakly_labeled_race_dataset.csv", split='train')
val_ds   = maskedABSA_Dataset("Dataset/weakly_labeled_race_dataset.csv", split='val')
test_ds  = maskedABSA_Dataset("Dataset/weakly_labeled_race_dataset.csv", split='test')

num_aspects = train_ds.num_aspects

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = StanceBERT(num_aspects=num_aspects, model_name=MODEL_NAME).to(DEVICE)

# Optimizer & scheduler
no_decay = ["bias", "LayerNorm.weight"]
param_groups = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(param_groups, lr=LR)

num_training_steps = EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

from tqdm import tqdm

def run_epoch(loader, train: bool = True):
    model.train(train)
    if train:
        optimizer.zero_grad()

    total_loss, correct, total_known, total_batches = 0.0, 0.0, 0.0, 0

    # create progress bar
    mode = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"{mode} Epoch", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        word_one_hot = batch["word_one_hot"].to(DEVICE)
        stance = batch["stance"].to(DEVICE)
        known_mask = batch["known_mask"].to(DEVICE)

        with torch.set_grad_enabled(train):
            logits = model(input_ids, attention_mask, token_type_ids, word_one_hot)
            loss = masked_bce_loss(logits, stance, known_mask)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds[known_mask] == stance[known_mask]).float().sum().item()
            total_known += known_mask.sum().item()

            if train:
                (loss / GRAD_ACCUM_STEPS).backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

        total_loss += loss.item()
        total_batches += 1

        # update progress bar
        avg_loss = total_loss / total_batches
        avg_acc = (correct / total_known) if total_known > 0 else 0.0
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    # flush leftover grads
    if train and (len(loader) % GRAD_ACCUM_STEPS != 0):
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(1, total_batches)
    avg_acc = (correct / total_known) if total_known > 0 else 0.0
    return avg_loss, avg_acc



best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

    # simple checkpointing on best val loss
    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "model_state": model.state_dict(),
            "tokenizer_name": MODEL_NAME,
            "num_aspects": num_aspects,
            "unique_words_list": train_ds.unique_words_list
        }, "stance_bert.pt")
        print("✓ Saved checkpoint: stance_bert.pt")

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, correct, total_known = 0.0, 0.0, 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        word_one_hot = batch["word_one_hot"].to(DEVICE)
        stance = batch["stance"].to(DEVICE)
        known_mask = batch["known_mask"].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids, word_one_hot)
        loss = masked_bce_loss(logits, stance, known_mask)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds[known_mask] == stance[known_mask]).float().sum().item()
        total_known += known_mask.sum().item()

    acc = (correct / total_known) if total_known > 0 else 0.0
    return total_loss / max(1, len(loader)), acc

test_loss, test_acc = evaluate(test_loader)
print(f"[TEST] loss {test_loss:.4f} acc {test_acc:.4f}")

# -------- Inference helper --------
def predict_stance(text: str, one_hot: torch.Tensor, unique_words_list: List[str], threshold: float = 0.5):
    model.eval()
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).to(DEVICE)

    one_hot = one_hot.unsqueeze(0).to(DEVICE)  # (1, A)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids, one_hot)
        probs = torch.sigmoid(logits).squeeze(0)  # (A,)
        preds = (probs >= threshold).long()       # 1=pro, 0=anti

    # Only return stances for words present in the one-hot
    out = {}
    for i, present in enumerate(one_hot.squeeze(0).tolist()):
        if present == 1.0:
            out[unique_words_list[i]] = ("pro", float(probs[i])) if preds[i] == 1 else ("anti", float(probs[i]))
    return out

# Example:
# one_hot_vec = torch.zeros(num_aspects)
# for w in ["trump", "immigration", "economy"]:
#     if w in train_ds.unique_words_list:
#         one_hot_vec[train_ds.unique_words_list.index(w)] = 1.0
# print(predict_stance("I like the strong economy but disagree with Trump's rhetoric.", one_hot_vec, train_ds.unique_words_list))
