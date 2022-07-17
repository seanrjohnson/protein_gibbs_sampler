import esm
from typing import Sequence, Tuple, List, Union
import torch
RawMSA = Sequence[Tuple[str, str]]

def rawbatchlen(raw_batch: str):
    count = 0
    counting = True
    for ch in raw_batch:
        if ch == "<":
            counting = False
        if ch == ">":
            counting = True
        if counting == True:
            count += 1
    return count

class MSABatchConverter(esm.data.BatchConverter):
    """
    Monkeypatched from esm.data.MSABatchConverter
    """
    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(rawbatchlen(msa[0][1]) for msa in raw_batch)
        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []
        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(rawbatchlen(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens
        return labels, strs, tokens

esm.data.MSABatchConverter = MSABatchConverter

# based on examples from here: https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
class ESM1b():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM1v():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM6():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM12():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM34():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM_MSA1():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        # self.msa_transformer = model.eval().cuda()
        self.batch_converter = self.alphabet.get_batch_converter()