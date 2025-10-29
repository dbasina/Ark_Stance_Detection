import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int, get_config
from dataloader import  *

import torch
from engine import ark_engine
from utils import get_config
from utils import collate_fn_masked
from models import StanceBERT_Masked



collate_fn = collate_fn_masked
datasets_config = get_config('datasets_config.yaml')
dataset = MaskedABSA_politic(file_path = datasets_config['MaskedABSA_politic']['data_dir'], split='val', aspects_list = datasets_config['MaskedABSA_politic']['aspect_list'], annotation_percent=100)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)

stance_list = [2]
model = StanceBERT_Masked(stance_list)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
count = 0
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    token_type_ids = batch['token_type_ids']
    stance = batch['stance']
    mask_index = batch['mask_index']

    fused, logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_index=mask_index, head_n=0)
    loss = loss_fn(logits, stance)
    print("Loss:", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


