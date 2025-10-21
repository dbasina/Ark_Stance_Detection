# sample_infer_from_dataset.py
import argparse
import random
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from dataset import maskedABSA_Dataset
from models import StanceBERT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    required = ["model_state", "tokenizer_name", "num_aspects", "unique_words_list"]
    for k in required:
        if k not in ckpt:
            raise ValueError(f"Checkpoint missing key: {k}")
    return ckpt


def collate_fn(examples: List[Dict[str, Any]], tokenizer, max_len: int):
    texts = [ex["text"] for ex in examples]
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    word_one_hot = torch.stack([ex["word_one_hot_vector"] for ex in examples])  # (B,A)
    stance = torch.stack([ex["stance_vector"] for ex in examples])              # (B,A)
    known_mask = (word_one_hot == 1.0) & (stance != -1.0)

    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])),
        "word_one_hot": word_one_hot,
        "stance": stance,
        "known_mask": known_mask,
        # keep raw fields for pretty printing
        "texts": texts,
    }
    return batch


@torch.no_grad()
def predict_batch(model, batch, threshold: float):
    model.eval()
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    token_type_ids = batch["token_type_ids"].to(DEVICE)
    one_hot = batch["word_one_hot"].to(DEVICE)
    logits = model(input_ids, attention_mask, token_type_ids, one_hot)
    probs = torch.sigmoid(logits)  # (B,A)
    preds = (probs >= threshold).long()  # (B,A) 1=pro, 0=anti
    return probs.cpu(), preds.cpu()


def pretty_print_samples(texts, one_hot, probs, preds, stance_gold, known_mask, vocab):
    for i, text in enumerate(texts):
        print("\n" + "=" * 100)
        print("TEXT:")
        print(text)
        print("-" * 100)

        active_idxs = [j for j, v in enumerate(one_hot[i].tolist()) if v == 1.0]
        if not active_idxs:
            print("No target words (one-hot all zeros).")
            continue

        # header
        w_col = max(4, max(len(vocab[j]) for j in active_idxs))
        print(f"{'word'.ljust(w_col)} | pred_stance | prob_pro | gold (if known)")
        print("-" * (w_col + 36))

        for j in active_idxs:
            p = float(probs[i, j].item())
            pred = "pro" if preds[i, j].item() == 1 else "anti"
            gold = "-"
            if known_mask[i, j].item():
                gold = "pro" if stance_gold[i, j].item() == 1.0 else "anti"
            print(f"{vocab[j].ljust(w_col)} | {pred:^11} | {p:7.4f} | {gold}")


def main():
    parser = argparse.ArgumentParser(
        description="Run StanceBERT on samples from maskedABSA_Dataset."
    )
    parser.add_argument("--checkpoint", type=str, default="stance_bert.pt")
    parser.add_argument("--csv", type=str, default="Dataset/weakly_labeled_race_dataset.csv")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=5, help="How many samples to show")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1) Load checkpoint & restore tokenizer/model
    ckpt = load_checkpoint(args.checkpoint)
    model_name = ckpt["tokenizer_name"]
    num_aspects = ckpt["num_aspects"]
    vocab = ckpt["unique_words_list"]

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = StanceBERT(num_aspects=num_aspects, model_name=model_name).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)

    # 2) Load dataset split
    ds = maskedABSA_Dataset(args.csv, split=args.split)

    # Safety check: vocab alignment (dataset vs checkpoint)
    if ds.num_aspects != num_aspects or ds.unique_words_list != vocab:
        raise ValueError(
            "Unique-words list in dataset does not match checkpoint.\n"
            f"Dataset aspects: {ds.num_aspects}, Checkpoint aspects: {num_aspects}.\n"
            "Ensure you're using the same unique_words_list."
        )

    # 3) Pick N samples deterministically and build a DataLoader
    #    We’ll just take the first N; to randomize, shuffle indices here.
    n = min(args.num-samples if hasattr(args, 'num-samples') else args.num_samples, len(ds))
    # argparse converts with dash to underscore: use args.num_samples
    n = min(args.num_samples, len(ds))
    subset = [ds[i] for i in range(n)]

    loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, n),
        shuffle=False,
        collate_fn=lambda ex: collate_fn(ex, tokenizer, args.max_len),
        pin_memory=torch.cuda.is_available(),
    )

    # 4) Run inference and pretty print
    for batch in loader:
        probs, preds = predict_batch(model, batch, threshold=args.threshold)
        pretty_print_samples(
            texts=batch["texts"],
            one_hot=batch["word_one_hot"],
            probs=probs,
            preds=preds,
            stance_gold=batch["stance"],
            known_mask=batch["known_mask"],
            vocab=vocab,
        )


if __name__ == "__main__":
    main()
