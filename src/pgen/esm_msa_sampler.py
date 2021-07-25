import torch
import math
import random
from tqdm import trange
from pgen.esm_sampler import generate_step

ESM_MSA_ALLOWED_AMINO_ACIDS = "-ACDEFGHIKLMNPQRSTVWY"

class ESM_MSA_sampler():
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

        self.valid_aa_idx = sorted([self.model.alphabet.get_idx(tok) for tok in ESM_MSA_ALLOWED_AMINO_ACIDS])

    def untokenize_batch(self, batch): #TODO: maybe should be moved to the model class, or a model superclass?
        #convert tokens to AAs, but skip the first one, because that one is <cls>

        out_batch = list()
        for msa in batch:
            out_batch += ["".join([self.model.alphabet.get_tok(itm) for itm in seq[1:]]) for seq in msa]

        return out_batch


    def get_init_msa(self, seed_msa, max_len, batch_size = 1):
        """ Get initial msa by padding seed_seq with masks, and then tokenizing."""

        padded_msa = list()
        for i, seq in enumerate(seed_msa):

            seq = seq.upper()
            input_chars = {s for s in seq}
            valid_chars = {s for s in ESM_MSA_ALLOWED_AMINO_ACIDS}
            if not input_chars.issubset(valid_chars):
                raise (Exception("Invalid input character: " + ",".join(input_chars - valid_chars)))

            remaining_len = max_len - len(seq)
            seq = list(seq) #if input is a string, convert it to an array
            padded_msa.append( (str(i), seq + ["<mask>"] * remaining_len) )
        
        labels, strs, tokens = self.model.batch_converter([padded_msa] * batch_size)
        return tokens

    def generate(self, n_samples, seed_msa, batch_size=1, in_order=False, max_len=None, leader_length=0, leader_length_percent=None, top_k=0, temperature=None, num_iters=10,  burnin=float('inf'),
                            mask=True, num_positions=0, num_positions_percent=None, indexes=None, rollover_from_start=False, show_progress_bar=True):
        """ generate sequences

            n_samples: number of sequences to output
            seed_msa: protein msa to start from
            batch_size: how many copies of the seed msa to run at one time.
            in_order: if True then cycle through the positions in order, otherwise randomly select positions each iteration
            max_len: maximum size of each generated sequence. If None, then use the length of the longest input msa.
            leader_length: don't overwrite this many amino acids at the beginning of the sequence.
            leader_length_percent: if not None, then will set leader_length = int(len(seed_seq)*(leader_length_percent / 100))
            top_k: if >0, only sample from the top k most probable AAs
            temperature: 
            num_iters: how many times to run the forward loop for every batch. 
            burnin: during burn-in period, sample from full distribution; afterwards take argmax, set to 0 to never sample (always take best), or inf to always sample

            num_positions: generate new AAs for this many positions each iteration. If 0, then generate for all target positions each round.
            num_positions_percent: If not None, then set num_positions = int(len(seed_seq)*(num_positions_percent / 100))
            indexes: positions of the input sequence to modify. 1-indexed, if None then all positions after the leader.

            show_progress_bar: if True then show a progress bar corresponding to the number of batches that need to be processed. Default: True.

            #### Examples #####
            seed = ["MTSENPLLALREKISALDEKLLALLAERRE-AVEVGKAKLLS-RPVRDIDRERDLLERLITLGKAHHLDAHYITRLFQLIIEDSVL-QQALLQQH",
                    "MSEEENLKTCREKL---DDKIIKLLAERFKIAEAIGKYKAENGLQIYDPKRERDILEHLEKKAEAEGLDAKYIRELFKKIIELGKKYQLLKLKEK",
                    "MSQPNDLPSLRERIDALDRRLVALLAERAQTVHEVGRLKAERGLPPRDPAREARLLER---LGREAELDPHLAERLWQAMIAELIERHRRLLADR",
                    "MSDPDPLAAARERIKALDEQLLALLA---ACALEVGRLKATHGLPVRDPERERALLERLLAQGEALGLSPEETRRLFEILIEESRRRQTRLLEQD"
                    ]
            #TODO: add some specific examples
        """


        #TODO: repetition penalty, somehow?
        #TODO: add dilated sequential sampling, like sampling every third or fifth amino acid and then doing the whole protein in like 3 or 5 steps, or something like that.
        with torch.no_grad(): # I'm not sure if this no_grad is necessary or not, but it couldn't hurt!

            num_sequences = len(seed_msa)
            sequence_length = len(seed_msa[0])

            cuda = self.cuda
            sequences = []
            n_batches = math.ceil(n_samples / num_sequences / batch_size)

            if num_positions_percent is not None:
                num_positions = int(sequence_length*(num_positions_percent / 100))
            if num_positions < 0:
                num_positions = 0

            if leader_length_percent is not None:
                leader_length = int(sequence_length*(leader_length_percent / 100))
            if leader_length < 0:
                leader_length = 0

            if max_len is None:
                max_len = sequence_length

            for batch_n in trange(n_batches, disable=(not show_progress_bar)):

                # shape: (batch, sequences, sequence_len)
                batch = self.get_init_msa(seed_msa, max_len, batch_size)
                batch = batch.cuda() if cuda else batch

                indexes, last_i = self.calculate_indexes(indexes, leader_length, max_len, rollover_from_start)

                if num_positions > len(indexes):
                    num_positions = len(indexes)


                for ii in range(num_iters):
                    if num_positions > 0: #do some subset of positions
                        if in_order: #cycle through the indexes
                            next_i = last_i
                            last_i, target_indexes = self.get_target_index_in_order(batch_size, indexes, next_i,
                                                                                    num_positions, num_sequences)
                        else:
                            target_indexes = self.get_random_target_index(batch_size, indexes, num_positions,
                                                                          num_sequences)
                    else:
                        target_indexes = self.get_target_indexes_all_positions(batch_size, indexes, num_sequences)

                    if mask:
                        self.mask_target_indexes(batch, target_indexes)

                    # shape: (batch, sequences, sequence_len, alphabet_digits)
                    out = self.model.model(batch)["logits"]

                    for batch_index in range(batch_size):
                        for sequence_index in range(num_sequences):
                            for kk in target_indexes[batch_index][sequence_index]:
                                idx = generate_step(out[batch_index][sequence_index],
                                                    gen_idx=kk,
                                                    top_k=top_k,
                                                    temperature=temperature,
                                                    sample=(ii < burnin),
                                                    valid_idx=self.valid_aa_idx)
                                batch[batch_index][sequence_index][kk] = idx
                if batch_n == (n_batches - 1): #last batch, so maybe don't take all of them, just take enough to get to n_samples
                    sequences += self.untokenize_batch(batch)[0:n_samples - len(sequences)]
                else:
                    sequences += self.untokenize_batch(batch)
            return sequences

    def mask_target_indexes(self, batch, target_indexes):
        for batch_index in range(len(batch)):
            for sequence_index in range(len(batch[batch_index])):
                for kk in target_indexes[batch_index][sequence_index]:
                    batch[batch_index][sequence_index][kk] = self.model.alphabet.mask_idx

    def get_target_indexes_all_positions(self, batch_size, indexes, num_sequences):
        target_indexes = list()
        for b in range(batch_size):
            target_indexes.append([indexes] * num_sequences)
        return target_indexes

    def get_random_target_index(self, batch_size, indexes, num_positions, num_sequences):
        target_indexes = list()
        for b in range(batch_size):
            batch_indexes = list()
            for s in range(num_sequences):
                batch_indexes.append(random.sample(indexes, num_positions))
            target_indexes.append(batch_indexes)
        return target_indexes

    def get_target_index_in_order(self, batch_size, indexes, next_i, num_positions, num_sequences):
        target_indexes = list()
        sampled = 0
        indexes_per_sequence = list()
        while sampled < num_positions:
            sampled += 1
            next_i = (next_i + 1) % len(indexes)
            indexes_per_sequence.append(indexes[next_i])
        for b in range(batch_size):
            target_indexes.append([indexes_per_sequence] * num_sequences)
        last_i = next_i
        return last_i, target_indexes

    def calculate_indexes(self, indexes, leader_length, max_len, rollover_from_start):
        if indexes is None:
            indexes = list(range(1, max_len + 1))  # skip position 1, because that should be <cls>
            if not rollover_from_start:  # we rollover from the end of the leader sequence
                indexes = indexes[leader_length:]
                last_i = leader_length - 1
            else:
                last_i = -1
        else:
            last_i = -1
        return indexes, last_i


    def calculate_log_likelihood_msa(self, msa, target_index, with_masking=True, verbose=False):
        """
            msa: a list of protein sequence strings, each of the same length.
            target_index: the sequence in the msa to mask
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.

        """
        # TODO: Allow batching to calculate likelihoods for multiple sequences at a time (how does padding effect likelihoods for sequences shorter than the longest sequence, hopefully not at all).

        # Inspired by and borrowing code from:
        # https://github.com/facebookresearch/esm/blob/master/variant-prediction/predict.py

        # log_likelihood_sum = 0
        
        # batch = [(str(0), list(seq.upper())),]
        # _, _, tokens = self.model.batch_converter(batch)
        # range_start = 0
        # if self.model.alphabet.prepend_bos:
        #     range_start = 1
        
        # range_end = tokens.shape[1]
        # if self.model.alphabet.append_eos:
        #     range_end -= 1
        
        # assert len(seq) == len(list(range(range_start, range_end)))

        # with torch.no_grad():
        #     if with_masking:
        #         for idx in range(range_start, range_end):
        #             old_tok = tokens[0,idx].item()
        #             tokens[0,idx] = self.model.alphabet.mask_idx
        #             token_probs = torch.log_softmax(self.model.model(tokens)['logits'], dim=-1)
        #             if verbose:
        #                 print(f"{self.model.alphabet.all_toks[old_tok]}\t{token_probs[0,idx,old_tok]}")
        #                 print(" ".join([f"{x}:{token_probs[0,idx,self.model.alphabet.tok_to_idx[x]]}" for x in self.model.alphabet.all_toks]))
        #             log_likelihood_sum += token_probs[0,idx,old_tok]
        #             tokens[0,idx] = old_tok
        #     else: #no masking, so we just need to calculate a single forward pass on the unmasked model
        #         token_probs = torch.log_softmax(self.model.model(tokens)['logits'], dim=-1)
        #         for idx in range(range_start, range_end):
        #             log_likelihood_sum += token_probs[0,idx,tokens[0,idx].item()]

        # return float(log_likelihood_sum / len(seq))
