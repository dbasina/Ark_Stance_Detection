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