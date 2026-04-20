import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        # T5 uses the pad token id (0) as the decoder start token by convention.
        self.decoder_start_token_id = self.tokenizer.pad_token_id
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')

        with open(nl_path, 'r') as f:
            nl_lines = [line.strip() for line in f.readlines()]

        # Tokenize encoder inputs with a T5-style task prefix.
        self.encoder_inputs = []
        for nl in nl_lines:
            ids = tokenizer(
                "translate English to SQL: " + nl,
                truncation=True,
                max_length=512,
            )['input_ids']
            self.encoder_inputs.append(torch.tensor(ids, dtype=torch.long))

        # On train/dev we also have SQL targets; on test we don't.
        if split != 'test' and os.path.exists(sql_path):
            with open(sql_path, 'r') as f:
                sql_lines = [line.strip() for line in f.readlines()]

            self.decoder_inputs = []
            self.decoder_targets = []
            for sql in sql_lines:
                tgt_ids = tokenizer(
                    sql,
                    truncation=True,
                    max_length=512,
                )['input_ids']  # T5 tokenizer already appends </s>
                tgt = torch.tensor(tgt_ids, dtype=torch.long)

                # Decoder input = [decoder_start] + target[:-1]; decoder target = full tgt ending in </s>.
                dec_in = torch.cat([
                    torch.tensor([self.decoder_start_token_id], dtype=torch.long),
                    tgt[:-1],
                ])
                self.decoder_inputs.append(dec_in)
                self.decoder_targets.append(tgt)
        else:
            self.decoder_inputs = None
            self.decoder_targets = None

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        enc = self.encoder_inputs[idx]
        initial_dec = torch.tensor([self.decoder_start_token_id], dtype=torch.long)
        if self.decoder_inputs is not None:
            return enc, self.decoder_inputs[idx], self.decoder_targets[idx], initial_dec
        else:
            return enc, initial_dec

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_seqs = [b[0] for b in batch]
    decoder_in_seqs = [b[1] for b in batch]
    decoder_tgt_seqs = [b[2] for b in batch]
    initial_decoder_seqs = [b[3] for b in batch]

    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence(decoder_in_seqs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_tgt_seqs, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_seqs, dim=0)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_seqs = [b[0] for b in batch]
    initial_decoder_seqs = [b[1] for b in batch]

    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(initial_decoder_seqs, dim=0)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x