from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import yaml
from scipy import interpolate
from PIL import Image
import torch.nn as nn
import copy
from transformers import BertTokenizerFast
from typing import Dict, Any, List
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless runs (servers/venvs)
import matplotlib.pyplot as plt

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


class MetricLogger(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressLogger(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def unmasked_metric_AUROC(target, prediction, known_mask, nb_classes=14):
    outAUROC = []

    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()
    known_mask = known_mask.cpu().numpy()

    # iterate over columns and compute AUROC for each aspect
    # if none of the samples have a particular aspect, then we append nan to the outAUROC for that aspect/column.
    column_list = []
    for i in range(nb_classes):
        column_mask = known_mask[:, i]

        #If no samples have this aspect, append nan
        if not np.any(column_mask):
            outAUROC.append(np.nan)
            continue

        #If only one stance present for this aspect in targets, append nan sicne we cannot compute AUROC
        if np.unique(target[column_mask, i]).size < 2:
            outAUROC.append(np.nan)
            continue
        column_list.append(i)
        target_column = target[column_mask, i]
        prediction_column = prediction[column_mask, i]
        auc = roc_auc_score(target_column, prediction_column)
        outAUROC.append(auc)
    return column_list, outAUROC

def metric_AUROC(target, output, nb_classes=2):
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(nb_classes):
        outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC


def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cosine_anneal_schedule(t,epochs,learning_rate):
    T=epochs
    M=1
    alpha_zero = learning_rate

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)

# --------------------------------------------------------
# Copy from DINO https://github.com/facebookresearch/dino
# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def unmasked_bce_loss(logits, targets, known_mask):
    """
    logits: (B, A) raw logits
    targets: (B, A) with values in {0,1} for known, -1 for unknown
    known_mask: (B, A) boolean where we compute loss
    """
    if known_mask.sum() == 0:
        # no known labels in this batch; return zero loss to keep training moving
        return logits.new_tensor(0.0, requires_grad=True)

    # Prepare targets for BCE
    targets = torch.where(known_mask, targets, torch.zeros_like(targets))
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')  # sum then normalize by count of known labels
    loss = loss_fn(logits[known_mask], targets[known_mask])
    loss = loss / known_mask.sum().clamp(min=1)
    return loss

def unmasked_multilabel_accuracy(y_test, p_test, known_mask, thresh=0.5):
    # y_test: [-1,0,1], p_test: probs [0,1], known_mask: bool [N,A]
    m = known_mask.bool()

    # filter the known aspects in the ground truth
    y_true = (y_test[m] == 1)

    # in the filtered predictions, if the prediction >= 0.5 then pro else anti                
    y_pred = (p_test[m] >= thresh)

    if y_true.numel() == 0:
        return float('nan')
    return (y_true == y_pred).float().mean().item()

def stance_detect_accuracy(targets, predictions):
    """
    targets: (N, C) one-hot vectors or class indices
    predictions: (N, C) logits or probabilities
    """
    if targets.ndim == 2:  # one-hot -> class index
        targets = targets.argmax(dim=1)
    preds = predictions.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def collate_fn_unmasked(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    
    texts = [ex['text'] for ex in examples]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
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

def collate_fn_masked(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    texts = [ex['masked_text'] for ex in examples]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    
    stance_one_hot = torch.stack([ex['stance_vector'] for ex in examples])
    mask_id = tokenizer.mask_token_id
    mask_positions = enc['input_ids'].eq(mask_id)         

    # Ensure exactly one [MASK] per example
    counts = mask_positions.sum(dim=1)  # [B]
    if not torch.all(counts == 1):
        mask_tok = tokenizer.mask_token
        mask_id  = tokenizer.mask_token_id
        bad = (counts != 1).nonzero(as_tuple=False).squeeze(1).tolist()

        print(f"mask_token: {mask_tok}  mask_token_id: {mask_id}")

        for i in bad:
            ids = enc["input_ids"][i].tolist()                        # list[int]
            toks = tokenizer.convert_ids_to_tokens(ids)               # list[str]
            mask_locs = [j for j, tid in enumerate(ids) if tid == mask_id]

            print(f"\n--- Bad example idx={i} ---")
            print("texts[i]:", texts[i])
            print("counts[i]:", int(counts[i]))
            print("mask locations:", mask_locs)
            print("tokens:", toks)

        # Optional: show which are zero vs >1
        none_mask = (counts == 0).nonzero(as_tuple=False).squeeze(1).tolist()
        multi_mask = (counts > 1).nonzero(as_tuple=False).squeeze(1).tolist()
        print("\nNo-mask indices:", none_mask, " Multi-mask indices:", multi_mask)

        raise ValueError(f"Each example must contain exactly one {mask_tok}; bad indices: {bad}")
    
    
    mask_index = mask_positions.int().argmax(dim=1)

    if examples[0].keys().__contains__('word_one_hot_vector'):    
        word_one_hot = torch.stack([ex['word_one_hot_vector'] for ex in examples])
        batch = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])),
            "word_one_hot": word_one_hot,
            "stance": stance_one_hot,
            "mask_index": mask_index
            
        }
    else:
        batch = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])),
            "stance": stance_one_hot,
            "mask_index": mask_index  
        }
    return batch

def count_aspect_occurrences(dataset_list, aspect_list):
    aspect_postive_count = [0]*len(aspect_list)
    aspect_negative_count = [0]*len(aspect_list)

    # Count occurrences of each aspect with 'pro' and 'anti' stances
    for dataset in dataset_list:
        for sample in dataset:
            sample_word_vector = sample['word_one_hot_vector']
            word_index = sample_word_vector.tolist().index(1.0)
            stance_label = sample['stance_label']
            if stance_label == 'pro':
                aspect_postive_count[word_index] += 1
            elif stance_label == 'anti':
                aspect_negative_count[word_index] += 1
            else:
                raise ValueError("stance_label must be one of 'pro' or 'anti'")



    return aspect_postive_count, aspect_negative_count

def generate_plots(save_path: str, aspect_list, aspect_positive_count, aspect_negative_count, top_n_polarized: int = 25, top_mixed_for_stack: int = 30, also_scatter: bool = True):
    """
    Generate plots using pre-extracted counts.

    Parameters
    ----------
    save_path : str
        Directory to save plots.
    aspect_list : list[str]
        Aspect names in the same order as counts.
    aspect_positive_count : list[int]
        Pro label counts aligned to aspect_list.
    aspect_negative_count : list[int]
        Anti label counts aligned to aspect_list.
    top_n_polarized : int
        Number of aspects to show in the diverging bar (by |pro-anti|).
    top_mixed_for_stack : int
        Max number of mixed (pro>0 & anti>0) aspects to show in stacked chart (by total).
    also_scatter : bool
        If True, saves a pro-vs-anti scatter plot as well.
    """

    os.makedirs(save_path, exist_ok=True)

    # Build DataFrame from counts
    df = pd.DataFrame({
        "aspect": aspect_list,
        "pro": aspect_positive_count,
        "anti": aspect_negative_count
    })
    df["total"] = df["pro"] + df["anti"]
    # Polarization metrics
    df["diff"] = df["pro"] - df["anti"]
    df["abs_diff"] = df["diff"].abs()
    # Shares (avoid division by zero)
    safe_total = df["total"].replace(0, np.nan)
    df["pro_share"] = df["pro"] / safe_total
    df["anti_share"] = df["anti"] / safe_total

    # -------------------------------
    # 1) Diverging Bar: polarization
    # -------------------------------
    df_pol = df.sort_values("abs_diff", ascending=False).head(top_n_polarized).copy()
    # Colors: anti-heavy = red, pro-heavy = green
    colors = df_pol["diff"].apply(lambda x: "#d73027" if x < 0 else "#1a9850")

    plt.figure(figsize=(12, max(6, 0.35 * len(df_pol))))
    plt.barh(df_pol["aspect"], df_pol["diff"], color=colors, edgecolor="none")
    plt.axvline(0, color="black", lw=0.8)
    plt.gca().invert_yaxis()

    plt.title(f"Top {len(df_pol)} Most Polarized Aspects (Pro − Anti)")
    plt.xlabel("Pro − Anti (count difference)")
    plt.ylabel("Aspect")

    # Annotate with signed difference and (optional) totals
    max_span = max(1, df_pol["abs_diff"].max())
    pad = 0.02 * max_span
    for y, (val, tot) in enumerate(zip(df_pol["diff"], df_pol["total"])):
        ha = "left" if val > 0 else "right"
        x = val + (pad if val > 0 else -pad)
        plt.text(x, y, f"{val:+d}  (n={int(tot)})", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    diverging_path = os.path.join(save_path, f"polarization_diverging_top{len(df_pol)}.png")
    plt.savefig(diverging_path, dpi=200)
    plt.close()

    # ----------------------------------------------------
    # 2) Stacked proportion bars for mixed (pro & anti)
    # ----------------------------------------------------
    mixed = df[(df["pro"] > 0) & (df["anti"] > 0)].copy()
    # For readability, keep top by total
    mixed = mixed.sort_values("total", ascending=False).head(top_mixed_for_stack)

    # Horizontal stacked bars (0..1)
    plt.figure(figsize=(12, max(6, 0.35 * len(mixed))))
    # Replace NaNs (if any) to 0 for plotting
    mixed["pro_share"] = mixed["pro_share"].fillna(0)
    mixed["anti_share"] = mixed["anti_share"].fillna(0)

    # Plot pro then anti as stacked segments
    plt.barh(mixed["aspect"], mixed["pro_share"], label="Pro", edgecolor="none")
    plt.barh(mixed["aspect"], mixed["anti_share"], left=mixed["pro_share"], label="Anti", edgecolor="none")

    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    plt.xlabel("Share within aspect (Proportion)")
    plt.ylabel("Aspect")
    plt.title(f"Mixed Aspects: Pro vs Anti Distribution (Top {len(mixed)} by total)")
    plt.legend(loc="lower right")

    # Annotate exact shares and totals
    for y, (p, a, t) in enumerate(zip(mixed["pro_share"], mixed["anti_share"], mixed["total"])):
        # center of pro segment
        if p > 0.05:
            plt.text(p / 2, y, f"{p:.0%}", va="center", ha="center", fontsize=8, color="white")
        # center of anti segment
        if a > 0.05:
            plt.text(p + a / 2, y, f"{a:.0%}", va="center", ha="center", fontsize=8, color="white")
        # show total at end of bar
        plt.text(1.01, y, f"n={int(t)}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    mixed_stack_path = os.path.join(save_path, f"mixed_aspects_stacked_proportion_top{len(mixed)}.png")
    plt.savefig(mixed_stack_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ---------------------------------------
    # 3) (Optional) Pro vs Anti scatter plot
    # ---------------------------------------
    if also_scatter:
        plt.figure(figsize=(8, 7))
        # bubble size scales with total (tweak factor for visibility)
        sizes = np.clip(df["total"], 1, None)
        s = 50 * np.sqrt(sizes / sizes.max())  # gentle scaling

        plt.scatter(df["pro"], df["anti"], s=s, alpha=0.7)
        # diagonal reference
        lim = max(df["pro"].max(), df["anti"].max(), 1)
        plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)

        plt.xlabel("Pro labels")
        plt.ylabel("Anti labels")
        plt.title("Aspect Sentiment Scatter (size ∝ total)")
        plt.tight_layout()
        scatter_path = os.path.join(save_path, "pro_vs_anti_scatter.png")
        plt.savefig(scatter_path, dpi=200)
        plt.close()

    # Optionally, save a CSV snapshot for debugging / inspection
    df_out = df.sort_values(["abs_diff", "total"], ascending=[False, False])
    df_out.to_csv(os.path.join(save_path, "aspect_counts_with_metrics.csv"), index=False)

    print(f"[OK] Saved plots to:\n- {diverging_path}\n- {mixed_stack_path}" + (f"\n- {scatter_path}" if also_scatter else ""))
