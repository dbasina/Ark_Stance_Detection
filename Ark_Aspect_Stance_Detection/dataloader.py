import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
import os
from utils import get_config


class unMaskedABSA_race(Dataset):
    def __init__(self, file_path, split='train', aspects_list = None, annotation_percent=100):
        df = pd.read_csv(file_path)
        self.train_frac = 0.8
        self.val_frac = 0.1
        self.test_frac = 0.1

        self.unique_aspects_list = aspects_list
        self.num_aspects = len(self.unique_aspects_list)

        # --- 1) GROUP THE FULL DF BY TEXT FIRST ---
        grouped_all = []
        for text, group in df.groupby('text'):
            masked_text = group.iloc[0]['masked_text']

            word_one_hot_vector = torch.zeros(self.num_aspects, dtype=torch.float32)
            stance_vector = torch.full((self.num_aspects,), -1.0, dtype=torch.float32)

            for _, row in group.iterrows():
                word = str(row['word'])
                stance_label = str(row['stance']).strip().lower()

                if word not in self.unique_aspects_list:
                    # optional: continue or normalize casing/plurals here
                    continue

                idx = self.unique_aspects_list.index(word)
                word_one_hot_vector[idx] = 1.0

                if stance_label == 'pro':
                    stance_vector[idx] = 1.0
                elif stance_label == 'anti':
                    stance_vector[idx] = 0.0
                else:
                    stance_vector[idx] = -1.0

            grouped_all.append({
                'text': text,
                'masked_text': masked_text,
                'word_one_hot_vector': word_one_hot_vector,
                'stance_vector': stance_vector
            })

        # --- 2) SHUFFLE AT THE TEXT-LEVEL, THEN SPLIT ---
        rng = pd.Series(range(len(grouped_all))).sample(frac=1.0, random_state=42).tolist()
        grouped_all = [grouped_all[i] for i in rng]

        n = len(grouped_all)
        n_train = int(self.train_frac * n)
        n_val = int(self.val_frac * n)

        if split == 'train':
            data_split = grouped_all[:n_train]
        elif split == 'val':
            data_split = grouped_all[n_train:n_train + n_val]
        elif split == 'test':
            data_split = grouped_all[n_train + n_val:]
        else:
            raise ValueError("split must be one of 'train','val','test'")

        # --- 3) OPTIONAL ANNOTATION SUBSAMPLING ON THE TEXT-LEVEL ---
        if annotation_percent < 100:
            reduced_size = max(1, int((annotation_percent / 100) * len(data_split)))
            data_split = pd.Series(data_split).sample(n=reduced_size, random_state=42).tolist()

        self.data = data_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MaskedABSA_race(Dataset):
    def __init__(self, file_path, split='train', aspects_list = None, annotation_percent=100):
        df = pd.read_csv(file_path)

        self.num_stances = 2 # pro, anti
        self.stance_indicies = {'pro': 0, 'anti': 1}

        self.unique_aspects_list = aspects_list
        self.num_aspects = len(self.unique_aspects_list)

        self.train_frac = 0.8
        self.val_frac = 0.1
        self.test_frac = 0.1

        

        rng = random.Random(42)
        records = []
        for _, row in df.iterrows():
            # set text, masked_text
            
            
            text = row.get('text', "")
            masked_text = row.get('masked_text', "")
            
            # set word, word_one_hot_vector
            word_one_hot_vector = torch.zeros(self.num_aspects, dtype=torch.float32)
            word = str(row.get('word', ""))
            idx = self.unique_aspects_list.index(word)
            word_one_hot_vector[idx] = 1.0

            # set stance_label and stance_one_hot_vector
            stance_one_hot_vector = torch.zeros(self.num_stances, dtype=torch.float32)
            stance_label = str(row.get('stance')).strip().lower()
            if stance_label == 'pro':
                stance_one_hot_vector[self.stance_indicies['pro']] = 1.0
            elif stance_label == 'anti':
                stance_one_hot_vector[self.stance_indicies['anti']] = 1.0

            records.append({
                'text': text,
                'masked_text': masked_text,
                'word_one_hot_vector': word_one_hot_vector,
                'stance_label': stance_label,
                'stance_vector': stance_one_hot_vector
            })

        rng.shuffle(records)
        n = len(records)
        n_train = int(self.train_frac * n)
        n_val = int(self.val_frac * n)

        if split == 'train':
            data_split = records[:n_train]
        elif split == 'val':
            data_split = records[n_train:n_train + n_val]
        elif split == 'test':
            data_split = records[n_train + n_val:]
        else:
            raise ValueError("split must be one of 'train','val','test'")

        # define annotation_percent
        if annotation_percent < 100:
            reduced_size = max(1, int((annotation_percent / 100) * len(data_split)))
            data_split = rng.sample(data_split, reduced_size)

        self.data = data_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MaskedABSA_politic(Dataset):
    def __init__(self, file_path, split='train', aspects_list = None, annotation_percent=100):
        train_input = os.path.join(file_path,'politic_train_small.input')
        train_output= os.path.join(file_path,'politic_train_small.output')
        val_input = os.path.join(file_path,'politic_val_small.input')
        val_output= os.path.join(file_path,'politic_val_small.output')
        test_input = os.path.join(file_path,'politic_test_small.input')
        test_output= os.path.join(file_path,'politic_test_small.output')

        if split == 'train':
            input_file = train_input
            output_file= train_output
        elif split == 'val':
            input_file = val_input
            output_file= val_output
        elif split == 'test':
            input_file = test_input
            output_file= test_output
        else:
            raise ValueError("split must be one of 'train','val','test'")
        
        with open(input_file, 'r') as f:
            input_lines = [line.rstrip("\r\n") for line in f]
        with open(output_file, 'r') as f:
            output_lines = [line.rstrip("\r\n") for line in f]

        # convert output_lines to one-hot vectors\
        self.stance_indices = {'negative': 0, 'positive': 1}
        self.num_stances = len(self.stance_indices)

        stance_vectors = []
        for line in output_lines:
            if line not in ['negative', 'positive']:
                raise ValueError("stance must be one of 'negative','positive'")
            
            # set stance_label and stance_one_hot_vector
            stance_one_hot_vector = torch.zeros(self.num_stances, dtype=torch.float32)
            
            if line == 'negative':
                stance_one_hot_vector[self.stance_indices['negative']] = 1.0
            elif line == 'positive':
                stance_one_hot_vector[self.stance_indices['positive']] = 1.0
            stance_vectors.append(stance_one_hot_vector)

        self.stances = ['negative', 'positive']
        self.num_stances = len(self.stances)
        
        data = []
        for text, lab in zip(input_lines, output_lines):
            lab_norm = lab.strip().lower()
            if lab_norm not in self.stance_indices:
                raise ValueError(f"Invalid stance '{lab}'. Expected one of {list(self.stance_indices.keys())}")
            vec = torch.zeros(self.num_stances, dtype=torch.float32)
            vec[self.stance_indices[lab_norm]] = 1.0
            data.append({'masked_text': text, 'stance_vector': vec})

        # Optional downsampling by percent (deterministic slice)
        if not (1 <= annotation_percent <= 100):
            raise ValueError("annotation_percent must be in [1, 100]")
        k = max(1, round(len(data) * (annotation_percent / 100.0)))
        self.data = data[:k]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    

dict_dataloarder = {
    "unMaskedABSA_race" : unMaskedABSA_race,
    "MaskedABSA_race"   : MaskedABSA_race,
    "MaskedABSA_politic": MaskedABSA_politic
}



