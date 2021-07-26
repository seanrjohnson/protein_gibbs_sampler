import torch
import math
import random
from tqdm import trange

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, valid_idx=None):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
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
        #set device
        #TODO: handle case where there are multiple cuda devices.
        if (device == "gpu"):
            if (torch.cuda.is_available()):
                device = 'cuda:0'
                self.cuda = True
            else:
                raise(Exception("gpu requested, but No Cuda devices found"))
        else:
            device = "cpu"
        device = torch.device(device)
        self.model.model.to(device)

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



        if isinstance(seed_seq, list):
            batch = random.choices(seed_seq, k=batch_size)
            for i, seed in enumerate(batch):
                remaining_len = max_len - len(seed)
                batch[i] = (str(i), list(self.clean_seed_seq(seed)) + ["<mask>"] * remaining_len)

        elif isinstance(seed_seq, str):
            remaining_len = max_len - len(seed_seq)
            seed_seq = [x for x in self.clean_seed_seq(seed_seq)] #if input is a string, convert it to an array
            batch = [(str(i), seed_seq + ["<mask>"] * remaining_len) for i in range(batch_size)]

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
            burnin: during burn-in period, sample from full distribution; afterwards take argmax, set to 0 to never sample (always take best), or inf to always sample

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
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=False, num_positions=int(len(seed)/10), num_iters=15, burnin=5, mask=False)
            
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
                batch = batch.cuda() if cuda else batch

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


    def log_likelihood(self, seq, with_masking=True, verbose=False):
        """
            seq: a protein sequence string
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.

        """
        # TODO: Allow batching to calculate likelihoods for multiple sequences at a time (how does padding effect likelihoods for sequences shorter than the longest sequence, hopefully not at all).

        # Inspired by and borrowing code from:
        # https://github.com/facebookresearch/esm/blob/master/variant-prediction/predict.py

        log_likelihood_sum = 0
        
        batch = [(str(0), list(self.clean_seed_seq(seq)) ),]
        _, _, tokens = self.model.batch_converter(batch)
        range_start = 0
        if self.model.alphabet.prepend_bos:
            range_start = 1
        
        range_end = tokens.shape[1]
        if self.model.alphabet.append_eos:
            range_end -= 1
        
        assert len(seq) == len(list(range(range_start, range_end)))
        tokens = tokens.cuda() if self.cuda else tokens
        with torch.no_grad():
            if with_masking:
                for idx in range(range_start, range_end):
                    old_tok = tokens[0,idx].item()
                    tokens[0,idx] = self.model.alphabet.mask_idx
                    token_probs = torch.log_softmax(self.model.model(tokens)['logits'], dim=-1)
                    if verbose:
                        print(f"{self.model.alphabet.all_toks[old_tok]}\t{token_probs[0,idx,old_tok]}")
                        print(" ".join([f"{x}:{token_probs[0,idx,self.model.alphabet.tok_to_idx[x]]}" for x in self.model.alphabet.all_toks]))
                    log_likelihood_sum += token_probs[0,idx,old_tok]
                    tokens[0,idx] = old_tok
            else: #no masking, so we just need to calculate a single forward pass on the unmasked model
                token_probs = torch.log_softmax(self.model.model(tokens)['logits'], dim=-1)
                for idx in range(range_start, range_end):
                    log_likelihood_sum += token_probs[0,idx,tokens[0,idx].item()]

        return float(log_likelihood_sum / len(seq))
