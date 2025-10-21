
import torch
import torch.nn as nn

def masked_bce_loss(logits, targets, known_mask):
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

@torch.no_grad()
def masked_accuracy(logits, targets, known_mask, threshold=0.5):
    if known_mask.sum() == 0:
        return 0.0
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    correct = (preds[known_mask] == targets[known_mask]).float().sum()
    acc = (correct / known_mask.sum()).item()
    return acc

