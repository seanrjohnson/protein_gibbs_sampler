from typing import Iterator, List, Tuple
import torch
import math
import random
from tqdm import trange
from pgen.esm_sampler import generate_step
import sys

ESM_MSA_ALLOWED_AMINO_ACIDS = "-ACDEFGHIKLMNPQRSTVWY"
ESM_MSA_GAP_CHARACTERS = "-"  # there might be some reason to add "." to both of these constants some day.

def partition(input_list, num_partitions):
    if len(input_list) < num_partitions:
        num_partitions = len(input_list)

    out = [[] for x in range(num_partitions)]
    
    num_per_partition = len(input_list) // num_partitions
    remainder = len(input_list) % num_partitions
    
    target_seqs_per_partition = [num_per_partition] * num_partitions
    for i in range(remainder):
        target_seqs_per_partition[i] += 1
    list_i = 0
    for v in input_list:
        out[list_i].append(v)
        if (len(out[list_i]) == target_seqs_per_partition[list_i]):
            list_i += 1
    
    return out



class ESM_MSA_sampler():
    """adapted from bert-gen bert-babble.ipynb"""

    def __init__(self, model, device="cpu"):
        """
            model should be an object with parameters model, alphabet, and batch_converter
        """
        self.model = model

        #switch model to eval mode
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
        self.toks = [self.model.alphabet.get_tok(idx) for idx in self.valid_aa_idx]

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

            seq = self.clean_seed_seq(seq)

            remaining_len = max_len - len(seq)
            padded_msa.append((str(i), seq + "<mask>" * remaining_len))

        labels, strs, tokens = self.model.batch_converter([padded_msa] * batch_size)
        return tokens


    def clean_seed_seq(self, seq):
        seq = seq.upper()
        input_chars = {s for s in seq}
        valid_chars = {s for s in ESM_MSA_ALLOWED_AMINO_ACIDS}
        if not input_chars.issubset(valid_chars):
            raise (Exception("Invalid input character: " + ",".join(input_chars - valid_chars)))
        return seq

    # def probs_single(self, seed_msa, steps=10, target_index=-1, show_progress_bar=False):
    #     """
    #         calculate_probability for each position of an input sequence. runs one pass over 
    #         seed_msa: a list of sequences, they should be aligned and all the same length. The sequence at target_index in the list will be masked and sampled.
    #         steps: The positions in the sampled sequence will be randomly split into this many parts and they will be masked at the same time.
    #         target_index: index of the sequence to mask and sample.
    #         show_progress_bar: if True then a progress bar will be updated in the console.
        
    #         returns matrix, alphabet
            
    #         where:
    #             matrix is (alphabet_size * sequence size) where values are probabilities
    #             and
    #             alphabet is a list of len(alphabet_size) where values are the alphabet symbol at the corresponding row in the output matrix

    #     """
    #     sequence_length = len(seed_msa[0])
    #     out = torch.zeros((len(self.valid_aa_idx), sequence_length))

    #     with torch.no_grad():
    #         batch_index = 0
    #         cuda = self.cuda

    #         positions = list(range(1,sequence_length+1)) #shift by 1 to account for cls token at beginning of sequence

    #         batch = self.get_init_msa(seed_msa, len(seed_msa[0]), 1)
    #         batch = batch.cuda() if cuda else batch
    #         original_indexes = batch[batch_index][target_index].clone().detach()

    #         random.shuffle(positions)
    #         step_indices = partition(positions, steps)
            
    #         for step_i in trange(len(step_indices), disable=(not show_progress_bar)): # a step is one forward call of the model, where a subset of the sequence has been masked
    #             self.mask_target_indexes_single(batch, step_indices[step_i], -1)
    #             # shape: (batch, sequences, sequence_len, alphabet_digits)
    #             forward_pass = self.model.model(batch)["logits"]
                
    #             for kk, aa_position in enumerate(step_indices[step_i]): #position
    #                 #torch.distributions.categorical.Categorical(logits=kth_vals)
    #                 probs = torch.distributions.categorical.Categorical(logits=forward_pass[batch_index][target_index][aa_position][self.valid_aa_idx]).probs
    #                 out[:,aa_position-1] = probs

    #             batch[batch_index][target_index] = original_indexes
                

                
        
    #     self.untokenize_batch(batch)[target_index]
    #     return out.numpy(), self.toks

    def generate_single(self, seed_msa, steps=10, passes=3, burn_in=1, target_index=-1, k=1):
        """
            generate a single sequence from an MSA
            seed_msa: a list of sequences, they should be aligned and all the same length. The sequence at target_index in the list will be masked and sampled.
            steps: in every pass, the positions in the sampled sequence will be randomly split into this many parts and they will be masked at the same time.
            passes: how many complete passes to make over the sampled sequence.
            burn_in: sample from the complete distribution for this many passes, then only take the highest probability amino acid for the remaining passes.
            target_index: index of the sequence to mask and sample.
        """
        with torch.no_grad():
            sequence_length = len(seed_msa[0])
            cuda = self.cuda

            positions = list(range(1,sequence_length+1)) #shift by 1 to account for cls token at beginning of sequence

            batch = self.get_init_msa(seed_msa, len(seed_msa[0]), 1)
            batch = batch.cuda() if cuda else batch

            for pass_num in range(passes): # a pass is a complete pass over the sequence
                random.shuffle(positions)
                step_indices = partition(positions, steps)
                for step_i in range(len(step_indices)): # a step is one forward call of the model, where a subset of the sequence has been masked
                    #print(step_indices[step_i])
                    self.mask_target_indexes_single(batch, step_indices[step_i], -1)
                    #print(self.untokenize_batch(batch))
                    # shape: (batch, sequences, sequence_len, alphabet_digits)
                    forward_pass = self.model.model(batch)["logits"]
                    batch_index = 0
                    for aa_position in step_indices[step_i]: #position
                        idx = generate_step(forward_pass[batch_index][target_index],
                            gen_idx=aa_position, 
                            top_k=k, 
                            temperature=None,
                            sample=(pass_num < burn_in),
                            valid_idx=self.valid_aa_idx)
                        batch[batch_index][target_index][aa_position] = idx
                    
        return self.untokenize_batch(batch)[target_index]
        
    

    def generate(self, n_samples, seed_msa, batch_size=1, in_order=False, max_len=None, leader_length=0,
                 leader_length_percent=None, top_k=0, temperature=None, num_iters=10, burnin=float('inf'),
                 mask=True, num_positions=0, num_positions_percent=None, indexes=None, rollover_from_start=False,
                 show_progress_bar=True):
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
            burnin: during burn-in period, sample from full distribution; afterwards sample from top_k, set to 0 to never sample from full distribution (always take from top_k), or inf to always sample from full distribution.

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
        #      Like we do for the likelihood.
        with torch.no_grad(): # I'm not sure if this no_grad is necessary or not, but it probably doesn't hurt.

            num_sequences = len(seed_msa)
            sequence_length = len(seed_msa[0])

            cuda = self.cuda
            sequences = []
            n_generation_rounds = math.ceil(n_samples / num_sequences / batch_size)

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

            for generation_round in trange(n_generation_rounds, disable=(not show_progress_bar)):

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

                    for batch_index in range(batch_size): #msa
                        for sequence_index in range(num_sequences): #sequence
                            for kk in target_indexes[batch_index][sequence_index]: #position
                                
                                idx = generate_step(out[batch_index][sequence_index],
                                                    gen_idx=kk, # +1 is because of start token
                                                    top_k=top_k,
                                                    temperature=temperature,
                                                    sample=(ii < burnin),
                                                    valid_idx=self.valid_aa_idx)
                                batch[batch_index][sequence_index][kk] = idx
                if generation_round == (n_generation_rounds - 1): #last batch, so don't take all of them, just take enough to get to n_samples
                    sequences += self.untokenize_batch(batch)[0:n_samples - len(sequences)]
                else:
                    sequences += self.untokenize_batch(batch)
            return sequences

    def mask_target_indexes(self, batch, target_indexes):
        for batch_index in range(len(batch)):
            for sequence_index in range(len(batch[batch_index])):
                for kk in target_indexes[batch_index][sequence_index]:
                    batch[batch_index][sequence_index][kk] = self.model.alphabet.mask_idx 
    
    def mask_target_indexes_single(self, batch, target_indexes, seq_index):
        for batch_index in range(len(batch)):
            for kk in target_indexes:
                batch[batch_index][seq_index][kk] = self.model.alphabet.mask_idx 

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

    def log_likelihood(self, msa, target_index=-1, with_masking=True, verbose=False,
                       count_gaps=False, mask_distance=float("inf")) -> Tuple[float,List[float]]:
        """
            msa: a list of protein sequence strings, each of the same length.
            target_index: the sequence in the msa to mask
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.
            count_gaps: if True, then likelihoods for positions that are gaps in the target sequence will not be included in the averaging.
            mask_distance: For optimization, when masking individual positions, the distance between masked positions in the same execution, by default only one position is masked per model call.
        """
        return next(self.log_likelihood_batch([msa], target_index, with_masking, verbose, count_gaps, mask_distance))

    #TODO: convert to iterator
    def log_likelihood_batch(self, msa_list, target_index=-1, with_masking=True, verbose=False,
                       count_gaps=False, mask_distance=float("inf"), batch_size=1) -> Iterator[Tuple[float,List[float]]]:
        """
            msa_list: a list of MSAs to calculate log_likelihood for. Each msa is a list of strings of the same length containing characters -ACDEFGHIKLMNPQRSTVWY
            target_index: the sequence in the msa to mask. default -1 (the last sequence in the msa)
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.
            verbose: if True then print debug information to stdout.
            count_gaps: if True, then likelihoods for positions that are gaps in the target sequence will not be included in the averaging.
            mask_distance: For optimization, when masking individual positions, the distance between masked positions in the same execution, by default only one position is masked per model call.
            
            
            batch_size: number of MSAs to run at once. default=1.
            
        """

        # Inspired by and borrowing code from:
        # https://github.com/facebookresearch/esm/blob/master/variant-prediction/predict.py

        gap_tokens = {self.model.alphabet.get_idx(x) for x in ESM_MSA_GAP_CHARACTERS}
        n_msas = len(msa_list)
        if batch_size is None:
            batch_size = n_msas

        reformatted_msas = [[(str(idx), self.clean_seed_seq(seq)) for idx, seq in enumerate(msa)] for msa in msa_list]
        _, _, tokens = self.model.batch_converter(reformatted_msas)

        range_start = 1 if self.model.alphabet.prepend_bos else 0
        end_modifier = -1 if self.model.alphabet.append_eos else 0

        msa_seq_lengths = [len(msa[target_index]) for msa in msa_list]
        msa_denominator = msa_seq_lengths.copy() #seq_len if counting gaps, non-gapped length of target sequence if not counting gaps

        if not count_gaps:
            for i in range(len(msa_list)):
                for g in ESM_MSA_GAP_CHARACTERS:
                    msa_denominator[i] -= msa_list[i][target_index].count(g)

        msa_range_end = [seq_len + range_start + end_modifier for seq_len in msa_seq_lengths]
        overall_range_end = tokens.shape[2] + end_modifier

        assert max(msa_seq_lengths) == len(range(range_start, overall_range_end))
        for msa_idx in range(n_msas):
            assert msa_seq_lengths[msa_idx] == len(range(range_start, msa_range_end[msa_idx])), msa_idx

        # each msa is a sample. If you need to mask an msa multiple different ways, then each alternative masking is a sample.
        # batch shape: (sample, 2 (tuple), sequence_len) 
        # tokens shape: (sample, sequences, sequence_len) = alphabet_digit
        tokens = tokens.cuda() if self.cuda else tokens

        with torch.no_grad():
            original_tokens = tokens[:, target_index].clone().detach()

            if with_masking:
                for msa_idx in range(n_msas):
                    likelihood_sum = 0.0
                    likelihood_list = list()
                    original_msas = reformatted_msas[msa_idx] #is it possible to copy a slice from tokens?
                    original_string = original_msas[target_index][1]

                    num_samples_for_this_msa = int(min(mask_distance, len(original_string)))

                    # I think you can do this with a pytorch/numpy broadcast of a slice from tokens
                    all_samples_for_this_msa = [original_msas.copy() for _ in range(num_samples_for_this_msa)]

                    masked_idx = set()
                    _, _, all_samples_for_this_msa_tokens = self.model.batch_converter(all_samples_for_this_msa)
                    # all_samples_for_this_msa_tokens = all_samples_for_this_msa_tokens.cuda() if self.cuda else all_samples_for_this_msa_tokens

                    for i_sample in range(num_samples_for_this_msa):
                        positions = range(range_start + i_sample, msa_range_end[msa_idx], num_samples_for_this_msa)
                        all_samples_for_this_msa_tokens[i_sample, target_index, positions] = self.model.alphabet.mask_idx
                        masked_idx.update(positions)
                    assert len(masked_idx) == len(original_string), sorted(masked_idx)

                    if verbose:
                        print(all_samples_for_this_msa_tokens[:, target_index])

                    counted_idx = set()
                    for batch_start in range(0,num_samples_for_this_msa, batch_size):

                        # tokens shape: (sample, sequences, sequence_len)
                        this_batch = all_samples_for_this_msa_tokens[batch_start:batch_start+batch_size,:,:]
                        this_batch = this_batch.cuda() if self.cuda else this_batch
                        token_probs = torch.log_softmax(self.model.model(this_batch)['logits'], dim=-1)


                        for i_sample in range(token_probs.shape[0]):
                            for idx_pos in range(range_start, msa_range_end[msa_idx]):
                                if (idx_pos-range_start) % num_samples_for_this_msa == (i_sample + batch_start):
                                    if count_gaps or original_tokens[msa_idx, idx_pos].item() not in gap_tokens: # only add the likelihood to the running sum if we are counting gaps, or if the position does not contain a gap.
                                        likelihood = token_probs[i_sample, target_index, idx_pos, original_tokens[msa_idx, idx_pos].item()]
                                        likelihood_sum += likelihood
                                        likelihood_list.append(likelihood.item())
                                        counted_idx.add(idx_pos)
                    assert len(counted_idx) == msa_denominator[msa_idx], sorted(counted_idx)

                    yield (float(likelihood_sum / msa_denominator[msa_idx]), likelihood_list)
            else:

                for batch_start in range(0, tokens.shape[0], batch_size):

                    token_probs = torch.log_softmax(
                        self.model.model(tokens[batch_start:batch_start + batch_size, :, :])['logits'], dim=-1)

                    for i_sample in range(token_probs.shape[0]):
                        log_likelihood_sum = 0.0
                        log_likelihood_list = []
                        for idx in range(range_start, msa_range_end[i_sample + batch_start]):
                            if count_gaps or original_tokens[i_sample + batch_start, idx].item() not in gap_tokens:  # only add the likelihood to the running sum if we are counting gaps, or if the position does not contain a gap.
                                likelihood = token_probs[i_sample, target_index, idx, original_tokens[i_sample + batch_start, idx].item()]
                                log_likelihood_sum += likelihood
                                log_likelihood_list.append(likelihood.item())
                        yield (float(log_likelihood_sum / msa_denominator[i_sample + batch_start]),log_likelihood_list)
