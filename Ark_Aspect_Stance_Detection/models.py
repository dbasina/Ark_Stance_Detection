import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from timm.models.helpers import load_state_dict
from transformers import BertModel

## StanceBERT_unMasked currently doesn't have full functionality implemented. 
class StanceBERT_unMasked(nn.Module):
    def __init__(self, num_aspects_list, hidden: int = 256):
        super().__init__()
        assert num_aspects_list is not None
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        self.relu_dropout = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        #multi-task heads
        self.aspect_proj = []
        self.omni_heads = []
        for num_aspects in num_aspects_list:
            self.aspect_proj.append(nn.Linear(num_aspects, hidden) if num_aspects > 0 else nn.Identity())
            self.omni_heads.append(nn.Linear(hidden_size + hidden, num_aspects) if num_aspects > 0 else nn.Identity())
        
        self.omni_heads = nn.ModuleList(self.omni_heads)
        self.aspect_proj = nn.ModuleList(self.aspect_proj)

    def forward(self, input_ids, attention_mask, token_type_ids, word_one_hot, head_n=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        asp = self.aspect_proj[head_n](word_one_hot)
        asp = self.relu_dropout(asp)
        fused = torch.cat([cls, asp], dim=-1)
        logits = self.omni_heads[head_n](fused)
        return fused, logits

class StanceBERT_Masked(nn.Module):
    def __init__(self, num_stances_list):
        super().__init__()
        assert num_stances_list is not None
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        self.relu_dropout = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        #multi-dataset heads
        self.omni_heads = []
        for num_stances in num_stances_list:
            self.omni_heads.append(nn.Linear(2*hidden_size, num_stances) if num_stances > 0 else nn.Identity())
        
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, input_ids, attention_mask, token_type_ids,mask_index, head_n=None):
        mask_index = mask_index.long()
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = out.last_hidden_state
        B,T,H = last_hidden_state.shape

        #Mask
        index = mask_index.view(B,1,1).expand(B,1,H)
        mask_vector = last_hidden_state.gather(1,index).squeeze(1)

        #Global Context
        cls = last_hidden_state[:, 0, :]

        # output
        fused = torch.cat([cls, mask_vector], dim=-1)
        logits = self.omni_heads[head_n](fused)
        return fused, logits


def build_omni_model_from_checkpoint(args, num_classes_list, key):
    
    if args.model_name == "stance_bert_unmasked":
        model = StanceBERT_unMasked(num_classes_list)
    elif args.model_name == "stance_bert_masked":
        model = StanceBERT_Masked(num_classes_list)
    else:
        raise NotImplementedError
    
    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights)
        state_dict = checkpoint[key]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))     
           
    return model

def build_omni_model(args, num_classes_list):

    if args.model_name == "stance_bert_unmasked":
        model = StanceBERT_unMasked(num_classes_list)
    elif args.model_name == "stance_bert_masked":
        model = StanceBERT_Masked(num_classes_list)
    else:
        raise NotImplementedError
 
    if args.pretrained_weights is not None:
        if args.pretrained_weights.startswith('https'):
            state_dict = load_state_dict_from_url(url=args.pretrained_weights, map_location='cpu')
        else:
            state_dict = load_state_dict(args.pretrained_weights)
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
      
        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))

    return model

def save_checkpoint(state,filename='model'):
    torch.save(state, filename + '.pth.tar')