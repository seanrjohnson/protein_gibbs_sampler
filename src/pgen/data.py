import torch
from sklearn.model_selection import train_test_split
from esm.data import FastaBatchedDataset
from .mlm import mask_sequence

def load_train_splits_from_fasta(datapath, batch_converter, args):
    dataset = DupeLabelAllowedFastaBatchedDataset.from_file(datapath)

    train_seq, val_seq, train_label, val_label = train_test_split(dataset.sequence_strs, dataset.sequence_labels, train_size=0.8, random_state=42)
    
    train_dataset = FastaBatchedDataset(train_label, train_seq)
    train_batches = train_dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=batch_converter, batch_sampler=train_batches
    )
    val_dataset = FastaBatchedDataset(val_label, val_seq)
    val_batches = val_dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=batch_converter, batch_sampler=val_batches
    )

    return dataset, train_dataloader, val_dataloader

class DupeLabelAllowedFastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(sequence_labels) == len(sequence_strs)
        # assert len(set(sequence_labels)) == len(sequence_labels)

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches

class MaskingBatchConverter:
    """ Wrap ESM batch converter with masking added """
    def __init__(self, alphabet, tokenizer):
        self.tokenizer = tokenizer
        self.alphabet = alphabet

    def __call__(self, raw_batch):
        # NOTE: RoBERTa uses EOS token while ESM-1 does not
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for label, seq_str in raw_batch) # Get max seq_str length
        
        # Setup containers
        # token container shape = (batch_size, max_len+include_bos+include_eos) # NOTE: No separator in the case of no 2nd task
        token_ids = torch.empty((batch_size, max_len + int(self.alphabet.prepend_bos) + \
            int(self.alphabet.append_eos)), dtype=torch.int64)
        token_ids.fill_(self.alphabet.padding_idx)
        raw_labels = []
        raw_seqs = []

        # Container for masked_token_labels
        # Automatically set all to -1, which will be used to mean to not include in loss later
        masked_lm_labels = torch.empty((batch_size, max_len + int(self.alphabet.prepend_bos) + \
            int(self.alphabet.append_eos)), dtype=torch.int64)
        masked_lm_labels.fill_(-1)
        

        # Iterate over raw batch of (label, seq) and create tokenized representations
        for i, (label, seq_str) in enumerate(raw_batch):
            # Track all raw labels, seqs
            raw_labels.append(label)
            raw_seqs.append(seq_str)

            # Optionally Add BOS token
            if self.alphabet.prepend_bos:
                token_ids[i, 0] = self.alphabet.cls_idx
            
            # Tokenize sequence
            tokenized_seq = self.tokenizer.tokenize(seq_str)

            # Apply Masking
            masked_seq, mask_labels = mask_sequence(tokenized_seq, self.alphabet)
            masked_seq = torch.tensor(masked_seq, dtype=torch.int64)
            mask_labels = torch.tensor(mask_labels, dtype=torch.int64)
            
            # Add masked sequence into indexed result
            token_ids[i, int(self.alphabet.prepend_bos):len(seq_str) + int(self.alphabet.prepend_bos)] = masked_seq
            masked_lm_labels[i, int(self.alphabet.prepend_bos):len(seq_str) + int(self.alphabet.prepend_bos)] = mask_labels
            
            # Optionally add EOS token
            if self.alphabet.append_eos:
                token_ids[i, len(seq_str) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return raw_labels, raw_seqs, token_ids, masked_lm_labels
