import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from project.utils.seed import set_seed
from project.utils.weighted_random_sampler import initialize_weighted_sampler

set_seed(42)


class CNCDataset(Dataset):
    def __init__(self, data_type, numeric_seqs, cat_seqs, labels, device):
        self.data_type = data_type
        self.numeric_seqs = [torch.tensor(seq, dtype=torch.float32, device=device) for seq in numeric_seqs]
        self.cat_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in cat_seqs]
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.labels)
    
    def get_data_type(self):
        return self.data_type

    def __getitem__(self, idx):
        numeric_seq = self.numeric_seqs[idx]
        cat_seq = self.cat_seqs[idx]
        label = self.labels[idx]
        length = numeric_seq.shape[0]
        return numeric_seq, cat_seq, label, length
    
    
def make_dataset(data_type, device):
    root = os.getcwd()
    data_dict = torch.load(os.path.join(root, 'project/data/preprocessed_lstm_data.pt'))
    
    numeric_seqs = data_dict[f'X_{data_type}_num']
    cat_seqs = data_dict[f'X_{data_type}_cat']
    labels = data_dict[f'y_{data_type}']
    
    dataset = CNCDataset(data_type, numeric_seqs, cat_seqs, labels, device)
    print(f'Loaded {data_type} data with {len(dataset)} samples.')
    return dataset
    
    
def collate_fn(batch):
    numeric_list, cat_list, labels, lengths = zip(*batch)
    
    numeric_padded = pad_sequence(numeric_list, batch_first=True, padding_value=0)
    cat_padded = pad_sequence(cat_list, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return numeric_padded, cat_padded, labels, lengths


def get_data_loader(dataset, batch_size, use_weighted_sampler=False):
    assert isinstance(dataset, CNCDataset), 'Dataset must be an instance of CNCDataset'
    if use_weighted_sampler:
        sampler = initialize_weighted_sampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)