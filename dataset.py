import torch
from torch.utils.data import Dataset
import pandas as pd

# Read unique_words in maskedABSA_Dataset class


import torch
from torch.utils.data import Dataset
import pandas as pd

class maskedABSA_Dataset(Dataset):
    def __init__(self, file_path, split='train', annotation_percent=100):
        df = pd.read_csv(file_path)

        self.train_frac = 0.8
        self.val_frac = 0.1
        self.test_frac = 0.1

        self.unique_words_list = ['white supremacists',
            'lie',
            'killer',
            'white supremacist',
            'mass shooter',
            'shooter',
            'violence',
            'border',
            'shooting',
            'human trafficking',
            'killing',
            'murderer',
            'lies',
            'terrorists',
            'murder',
            'crimes',
            'false narrative',
            'false flag',
            'terror',
            'psyops',
            'chokehold',
            'racist',
            'black americans',
            'african american',
            'democrats',
            'african americans',
            'systemic racism',
            'white americans',
            'institutional racism',
            'joe biden',
            'biden',
            'crime',
            'domestic terrorist',
            'racists',
            'democrat',
            'biden administration',
            'economy',
            'criminals',
            'segregation',
            'whites',
            'joe',
            'white race',
            'black vote',
            'white people',
            'democrat party',
            'newsom',
            'democratic party',
            'dem',
            'white women',
            'white folks',
            'white person',
            'apartheid',
            'patriot',
            'christian',
            'liberals',
            'hate crime',
            'republicans',
            'republican',
            'jane fonda',
            'christians',
            'climate change',
            'white men',
            'climate crisis',
            'school',
            'catholic',
            'schools',
            'university',
            'discrimination',
            'blm',
            'american',
            'mob',
            'justice',
            'americans',
            'black people',
            'domestic terrorists',
            'murders',
            'woke',
            'america',
            'patriots',
            'diversity',
            'inclusion',
            'family',
            'families',
            'dems',
            'white house',
            'border crisis',
            'corruption',
            'house',
            'american people',
            'protest',
            'bbc',
            'netflix',
            'race card',
            'trump',
            'president trump',
            'obama',
            'bill clinton',
            'bill',
            'kkk',
            'blacks',
            'confederacy',
            'antifa',
            'maga',
            'j6',
            'insurrection',
            'fbi',
            'girls',
            'hindus',
            'police brutality',
            'constitution',
            'riot',
            'rioters',
            'riots',
            '2020 riots',
            'protesters',
            'law',
            'terrorism',
            'antifa trans',
            'jesus',
            'faith',
            'illegals',
            'bidens',
            'justice system',
            'illegal immigrants',
            'dangerous terrorist threat',
            'white america',
            'lynching',
            'culture',
            'steve bannon',
            'democracy',
            'racist country',
            'injustice',
            'white nationalism',
            'domestic terrorism',
            'manufactured crisis',
            'maga republicans',
            'veterans',
            'education',
            'health care',
            'jobs',
            'student',
            'books',
            'kids',
            'book',
            'children',
            'child',
            'conservatives',
            'jordan neely',
            'white man',
            'black man',
            'death',
            'daniel penny',
            'homeless black man',
            'homeless',
            'jordan neelys',
            'marine veteran',
            'black men',
            'extremists',
            'disinformation',
            'sexual assault',
            'seditious conspiracy',
            'fascists',
            'mass shootings',
            'law enforcement',
            'genocide',
            'fascism',
            'extremism',
            'laws',
            'mom',
            'social security',
            'governor',
            'desantis',
            'liberty',
            'ron desantis',
            'magas',
            'mass shooting',
            'gun violence',
            'propaganda',
            'kid',
            'peace',
            'proof',
            'white media',
            'facts',
            'white supremacist beliefs',
            'evidence',
            'texas mall shooter',
            'republican party',
            'churches',
            'lives',
            'george floyd',
            'black lives',
            'racial justice',
            'black community',
            'deaths',
            'equality',
            'floyd',
            'george floyds',
            'michael brown',
            'protests',
            'black liberation organization',
            'black families',
            'gop',
            'medicare',
            'healthcare',
            'medicaid',
            'maga gop',
            'house republicans',
            'antiblack racism',
            'white nationalists',
            'freedom',
            'christianity',
            'civil rights',
            'kevin mccarthy',
            'mccarthy',
            'voters',
            'donald trump',
            'donald',
            'govt',
            'trumps',
            'tim scott',
            'baby',
            'life',
            'gops',
            'racial slurs',
            'misogyny',
            'vote',
            'tucker',
            'tucker carlson',
            'violent mob',
            'blm protester',
            'daniel perry',
            'march',
            'abbott',
            'greg abbott',
            'black children',
            'white mob',
            'black lives matter',
            'judge',
            'sexism']
        self.num_aspects = len(self.unique_words_list)

        # --- 1) GROUP THE FULL DF BY TEXT FIRST ---
        grouped_all = []
        for text, group in df.groupby('text'):
            masked_text = group.iloc[0]['masked_text']

            word_one_hot_vector = torch.zeros(self.num_aspects, dtype=torch.float32)
            stance_vector = torch.full((self.num_aspects,), -1.0, dtype=torch.float32)

            for _, row in group.iterrows():
                word = str(row['word'])
                stance_label = str(row['stance']).strip().lower()

                if word not in self.unique_words_list:
                    # optional: continue or normalize casing/plurals here
                    continue

                idx = self.unique_words_list.index(word)
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

def one_hot_to_word_list(one_hot_vector, unique_words_list):
    words = []
    for i, val in enumerate(one_hot_vector):
        if val.item() == 1.0:
            words.append(unique_words_list[i])
    return words

def stance_vector_to_dict(stance_vector, unique_words_list):
    stance_dict = {}
    for i, val in enumerate(stance_vector):
        if val.item() != -1.0:
            stance_dict[unique_words_list[i]] = 'pro' if val.item() == 1.0 else 'anti'
    return stance_dict


 




    
