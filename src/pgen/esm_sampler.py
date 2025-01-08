from typing import Iterator, List, Tuple
import torch
import math
import random
from tqdm import trange
import re

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, valid_idx=None):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Otherwise sample from top_k amino acids
        - valid_idx (list): list of valid indexes to return. If none, all indexes are valid
    returns:
        tensor containing the selected amino acid index
    """
    #TODO: repetition penalty.
    #TODO: this could be vectorized a lot better, but I think this isn't the rate limiting step (inferrence is), so it probably doesn't matter.

    logits = out[gen_idx] # 1 x vocab_size
    if temperature is not None:
        logits = logits / temperature

    if valid_idx is None:
        valid_idx = list(range(len(logits)))

    sub_logits = logits[valid_idx]

    if sample or (top_k <= 0) or (top_k > len(sub_logits)):
        # If sample is true, that means we are forcing sampling from the whole distribution.
        # If top_k is 0 that means we want to sample from the whole distribution.
        top_k = len(sub_logits)
    else:
        # top_k is in bounds and we aren't forcing full sampling, so just keep it as it is.
        top_k = top_k

    kth_vals, kth_idx = sub_logits.topk(top_k)  # kth_vals is the logits, kth_idx is the indexes at which the logits are found.
    dist = torch.distributions.categorical.Categorical(logits=kth_vals)

    idx = kth_idx[dist.sample()]

    return torch.tensor(valid_idx[idx])


ESM_ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

class ESM_sampler():
    """adapted from bert-gen bert-babble.ipynb"""


    def __init__(self, model, device="cpu"):
        """
            model should be an object with parameters model, alphabet, and batch_converter
        """
        self.model = model

        #switch model to eval mode
        #TODO: CHECK THAT THIS ACTUALLY SWITCHES THE MODEL TO EVAL MODE AND TURNS OFF GRADIENTS!
        self.model.model = self.model.model.eval()
        self.cuda = False
        self.device = device
        #set device
        if (self.device == "gpu"):
            self.device = "cuda:0"
        if re.match("^cuda:[0-9]+$", self.device):
            cuda_device_num = int(self.device.split(":")[1])
            if (torch.cuda.is_available()):
                self.cuda = True
            else:
                raise(Exception("gpu requested, but No Cuda devices found"))
            
            if (cuda_device_num >= torch.cuda.device_count()):
                raise(Exception("Invalid cuda device number: " + self.device))
        elif self.device != "cpu":
            raise(Exception("Invalid device: " + self.device))
        device = torch.device(self.device)
        self.model.model.to(self.device)

        self.valid_aa_idx = sorted([self.model.alphabet.get_idx(tok) for tok in ESM_ALLOWED_AMINO_ACIDS])

    def untokenize_batch(self, batch, bos, eos): #TODO: maybe should be moved to the model class, or a model superclass?
        #convert tokens to AAs, but skip the first one, because that one is <cls>
        start_offset = 0
        end_offset = 0
        if bos:
            start_offset = 1
        if eos:
            end_offset = -1
        out = [ "".join([self.model.alphabet.get_tok(seq[i]) for i in range(0 + start_offset, len(seq) + end_offset) ]) for seq in batch]
        return out

    @staticmethod
    def clean_seed_seq(seed_to_clean):
        cleaned_seq = seed_to_clean.upper()
        input_chars = {s for s in cleaned_seq}
        valid_chars = {s for s in ESM_ALLOWED_AMINO_ACIDS}
        if not input_chars.issubset(valid_chars):
            raise (Exception("Invalid input character: " + ",".join(input_chars-valid_chars)))
        return cleaned_seq

    def get_init_seq(self, seed_seq, max_len, batch_size = 1):
        """ Get initial sequence by padding seed_seq with masks """
        # In the BertGen paper they talk about padding with random sequence. I'm not sure that's a good idea. S.R.J.
        # Also, that code was commented out in the BertGen repo. So they probably didn't think was a good idea either.



        if isinstance(seed_seq, list): # input is an array, convert it to a string
            batch = random.choices(seed_seq, k=batch_size)
            for i, seed in enumerate(batch):
                remaining_len = max_len - len(seed)
                batch[i] = (str(i), self.clean_seed_seq(seed) + "<mask>" * remaining_len)

        elif isinstance(seed_seq, str):
            remaining_len = max_len - len(seed_seq)
            seed_seq = self.clean_seed_seq(seed_seq)
            batch = [(str(i), seed_seq + "<mask>" * remaining_len) for i in range(batch_size)]

        else:
            raise (Exception("seed sequence should either be a string or list"))

        labels, strs, tokens = self.model.batch_converter(batch)
        return tokens

    def generate(self, n_samples, seed_seq, batch_size=1, in_order=False, max_len=None, leader_length=0, leader_length_percent=None, top_k=0, temperature=None, num_iters=10,  burnin=float('inf'),
                            mask=True, num_positions=0, num_positions_percent=None, indexes=None, rollover_from_start=False, show_progress_bar=True):
        """ generate sequences

            n_samples: number of sequences to output
            seed_seq: protein msa to start from
            batch_size: how many copies of the seed msa to run at one time.
            in_order: if True then cycle through the positions in order, otherwise randomly select positions each iteration.
            max_len: maximum size of each generated sequence. If None, then use the length of the longest input msa.
            leader_length: don't overwrite this many amino acids at the beginning of the sequence.
            leader_length_percent: if not None, then will set leader_length = int(len(seed_seq)*(leader_length_percent / 100))
            top_k: if >0, only sample from the top k most probable AAs
            temperature: higher numbers will mean there is a lower penalty for low-scoring amino acids.
            num_iters: how many times to run the forward loop for every batch. 
            burnin: during burn-in period, sample from full distribution; afterwards sample from top_k, set to 0 to never sample from full distribution (always take from top_k), or inf to always sample from full distribution.

            num_positions: generate new AAs for this many positions each iteration. If 0, then generate for all target positions each round.
            num_positions_percent: If not None, then set num_positions = int(len(seed_seq)*(num_positions_percent / 100))
            indexes: positions of the input sequence to modify. 1-indexed, if None then all positions after the leader.

            show_progress_bar: if True then show a progress bar corresponding to the number of batches that need to be processed. Default: True.

            #### Examples #####
            seed = "MTSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLLSHRPVRDIDRERDLLERLITLGKAHHLDAHYITRLFQLIIEDSVLTQQALLQQH"

            #To generate AAs one position at a time in order:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, in_order=True, num_positions=1, num_iters=len(seed), mask=True)
            #To generate the entire protein at once:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=True, num_positions=len(seed), num_iters=1, mask=False)
            #To go 15 iterations over the protein where a 10% of AAs randomly distributed through the protein are mutated on each iteration:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=False, num_positions=int(len(seed)/10), num_iters=15, mask=False)
            #To go 15 iterations over the protein where a 10% of AAs randomly distributed through the protein are mutated on each iteration, and k=0 for the first 5 iterations, but k=1 for the remaining:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=False, num_positions=int(len(seed)/10), num_iters=15, burnin=5, k=1, mask=False)
            
            #### Sequence Completion ####
            seed = "MTSENPLLALREKISALDEKLLALLAERRELAVE"
            product_length = 95

            #generate L->R one at a time
                out = sampler.generate(1, seed_seq=seed, batch_size=1, max_len=product_length, in_order=True, top_k=0, leader_length=len(seed), num_positions=1, num_iters=product_length-len(seed), mask=True)
            #generate all at a time
                out = sampler.generate(1, seed_seq=seed, batch_size=1, max_len=product_length, in_order=True, top_k=0, leader_length=len(seed), num_positions=product_length-len(seed), num_iters=1, mask=True)
        """

        #TODO: repetition penalty, somehow?
        #TODO: add dilated sequential sampling, like sampling every third or fifth amino acid and then doing the whole protein in like 3 or 5 steps, or something like that.
        with torch.no_grad(): # I'm not sure if this no_grad is necessary or not, but it couldn't hurt!
            if isinstance(seed_seq, str):
                sequence_length = len(seed_seq)
            elif isinstance(seed_seq, list):
                sequence_length = max(len(seed) for seed in seed_seq)
            else:
                raise ValueError("Unknown seed sequence format, expecting str or list")

            cuda = self.cuda
            sequences = []
            n_batches = math.ceil(n_samples / batch_size)

            if max_len is None:
                max_len = sequence_length

            if num_positions_percent is not None:
                num_positions = int(max_len*(num_positions_percent / 100))
            if num_positions < 0:
                num_positions = 0

            if leader_length_percent is not None:
                leader_length = int(max_len*(leader_length_percent / 100))
            if leader_length < 0:
                leader_length = 0

            for batch_n in trange(n_batches, disable=(not show_progress_bar)):

                batch = self.get_init_seq(seed_seq, max_len, batch_size)
                batch = batch.cuda(self.device) if cuda else batch

                indexes, last_i = self.calculate_indexes(indexes, leader_length, max_len, rollover_from_start)

                if num_positions > len(indexes):
                    num_positions = len(indexes)

                for ii in range(num_iters):
                    if num_positions > 0: #do some subset of positions
                        if in_order: #cycle through the indexes
                            next_i = last_i
                            last_i, target_indexes = self.get_target_index_in_order(batch_size, indexes, next_i,
                                                                                    num_positions)
                        else:
                            target_indexes = self.get_random_target_index(batch_size, indexes, num_positions)
                    else:
                        target_indexes = [indexes] * batch_size

                    if mask:
                        self.mask_target_indexes(batch, target_indexes)

                    out = self.model.model(batch)["logits"]

                    for batch_index in range(batch_size):
                        for kk in target_indexes[batch_index]:
                            idx = generate_step(out[batch_index],
                                                gen_idx=kk,
                                                top_k=top_k,
                                                temperature=temperature,
                                                sample=(ii < burnin),
                                                valid_idx=self.valid_aa_idx)

                            batch[batch_index][kk] = idx

                if batch_n == (n_batches - 1): #last batch, so maybe don't take all of them, just take enough to get to n_samples
                    sequences += self.untokenize_batch(batch, self.model.alphabet.prepend_bos, self.model.alphabet.append_eos)[0:n_samples - len(sequences)]
                else:
                    sequences += self.untokenize_batch(batch, self.model.alphabet.prepend_bos, self.model.alphabet.append_eos)
            return sequences

    def get_random_target_index(self, batch_size, indexes, num_positions):
        target_indexes = list()
        for b in range(batch_size):
            target_indexes.append(random.sample(indexes, num_positions))
        return target_indexes

    def get_target_index_in_order(self, batch_size, indexes, next_i, num_positions):
        sampled = 0
        target_indexes = list()
        while sampled < num_positions:
            sampled += 1
            next_i = (next_i + 1) % len(indexes)
            target_indexes.append(indexes[next_i])
        target_indexes = [target_indexes] * batch_size
        last_i = next_i
        return last_i, target_indexes

    def mask_target_indexes(self, batch, target_indexes):
        for batch_index in range(len(target_indexes)):
            for kk in target_indexes[batch_index]:
                batch[batch_index][kk] = self.model.alphabet.mask_idx

    def calculate_indexes(self, indexes, leader_length, max_len, rollover_from_start):
        if indexes is None:
            indexes = range(1, max_len + 1)  # skip position 1, because that should be <cls>
            if not rollover_from_start:  # we rollover from the end of the leader sequence
                indexes = indexes[leader_length:]
                last_i = leader_length - 1
            else:
                last_i = -1
        else:
            last_i = -1
        return indexes, last_i


    def log_likelihood(self, seq, with_masking=True, verbose=False, mask_distance=float("inf"), batch_size=None) -> Tuple[float,List[float]]:
        """
            seq: a protein sequence string
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.
            mask_distance: For optimization, when masking individual positions, the distance between masked positions in the same execution, by default only one position is masked per model call.
            batch_size: number of MSAs to run on the gpu at once, if None, then batch_size=len(msa_list). default=None.
        """
        return next(self.log_likelihood_batch([seq], with_masking, verbose, mask_distance, batch_size))

    #TODO: convert to iterator
    def log_likelihood_batch(self, seq_list, with_masking=True, verbose=False, mask_distance=float("inf"), batch_size=None) -> Iterator[Tuple[float,List[float]]]:

        # TODO: Allow batching to calculate likelihoods for multiple sequences at a time (how does padding effect likelihoods for sequences shorter than the longest sequence, hopefully not at all).

        # Inspired by and borrowing code from:
        # https://github.com/facebookresearch/esm/blob/master/variant-prediction/predict.py

        n_batches = len(seq_list)
        if batch_size is None:
            batch_size = n_batches

        reformatted_seq = [(str(idx), self.clean_seed_seq(seq)) for idx, seq in enumerate(seq_list)]
        _, _, tokens = self.model.batch_converter(reformatted_seq)

        range_start = 1 if self.model.alphabet.prepend_bos else 0
        end_modifier = -1 if self.model.alphabet.append_eos else 0

        batch_range_end = [len(seq) + range_start for seq in seq_list]
        overall_range_end = tokens.shape[1] + end_modifier

        assert max(len(seq) for seq in seq_list) == len(range(range_start, overall_range_end))

        for b_idx in range(n_batches):
            assert len(seq_list[b_idx]) == len(range(range_start, batch_range_end[b_idx]))

        tokens = tokens.cuda(self.device) if self.cuda else tokens
        with torch.no_grad():
            if with_masking:
                old_toks = tokens.clone().detach()

                for seq_idx in range(len(reformatted_seq)):
                    likelihood_sum = 0.0
                    likelihood_list = list()

                    original_string = reformatted_seq[seq_idx][1]
                    num_sequences_to_process = int(min(mask_distance, len(original_string)))
                    all_samples_for_seq = [(idx, original_string) for idx, seq in enumerate(range(num_sequences_to_process))]

                    _, _, tokens_for_seq = self.model.batch_converter(all_samples_for_seq)

                    masked_idx = set()
                    for i_sample in range(num_sequences_to_process):
                        positions = range(range_start + i_sample, batch_range_end[seq_idx], num_sequences_to_process)
                        tokens_for_seq[i_sample, positions] = self.model.alphabet.mask_idx
                        masked_idx.update(positions)
                    assert len(masked_idx) == len(original_string), sorted(masked_idx)

                    counted_idx = set()
                    for batch_start in range(0, num_sequences_to_process, batch_size):

                        this_batch = tokens_for_seq[batch_start:batch_start + batch_size, :]
                        this_batch = this_batch.cuda(self.device) if self.cuda else this_batch
                        token_probs = torch.log_softmax(self.model.model(this_batch)['logits'], dim=-1)

                        for i_sample in range(token_probs.shape[0]):
                            for idx_pos in range(range_start, batch_range_end[seq_idx]):
                                if (idx_pos - range_start) % num_sequences_to_process == (i_sample + batch_start):
                                    likelihood = token_probs[i_sample, idx_pos, old_toks[seq_idx, idx_pos].item()]
                                    likelihood_sum += likelihood
                                    likelihood_list.append(likelihood.item())
                                    counted_idx.add(idx_pos)

                    assert len(counted_idx) == len(seq_list[seq_idx]), sorted(counted_idx)

                    yield (float(likelihood_sum / len(seq_list[seq_idx])), likelihood_list)

            else:  # no masking, so we just need to calculate a single forward pass on the unmasked model
                token_probs = torch.log_softmax(self.model.model(tokens)['logits'], dim=-1)
                for batch_idx in range(n_batches):
                    log_likelihood_sum = 0.0
                    log_likelihood_list = []
                    for idx in range(range_start, batch_range_end[batch_idx]):
                        likelihood = token_probs[batch_idx, idx, tokens[batch_idx, idx].item()]
                        log_likelihood_sum += likelihood
                        log_likelihood_list.append(likelihood.item())
                    yield (float(log_likelihood_sum / len(seq_list[batch_idx])), log_likelihood_list)
