import torch
import math
import random
from tqdm import trange
from pgen.esm_sampler import generate_step
import sys

ESM_MSA_ALLOWED_AMINO_ACIDS = "-ACDEFGHIKLMNPQRSTVWY"
ESM_MSA_GAP_CHARACTERS = "-"  # there might be some reason to add "." to both of these constants some day.

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

            seq = self.clean_seed_seq(seq)

            remaining_len = max_len - len(seq)
            seq = list(seq)  # if input is a string, convert it to an array
            padded_msa.append((str(i), seq + ["<mask>"] * remaining_len))

        labels, strs, tokens = self.model.batch_converter([padded_msa] * batch_size)
        return tokens


    def clean_seed_seq(self, seq):
        seq = seq.upper()
        input_chars = {s for s in seq}
        valid_chars = {s for s in ESM_MSA_ALLOWED_AMINO_ACIDS}
        if not input_chars.issubset(valid_chars):
            raise (Exception("Invalid input character: " + ",".join(input_chars - valid_chars)))
        return seq

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

    def log_likelihood(self, msa, target_index=-1, with_masking=True, verbose=False,
                       mask_entire_sequence=False, count_gaps=False, mask_distance=float("inf")):
        """
            msa: a list of protein sequence strings, each of the same length.
            target_index: the sequence in the msa to mask
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.
            mask_entire_sequence: if True, mask entire sequence instead of iterating over each position
            count_gaps: if True, then likelihoods for positions that are gaps in the target sequence will not be included in the averaging.
            mask_distance: For optimization, when masking individual positions, the distance between masked positions in the same execution, by default only one position is masked per model call.
        """
        return self.log_likelihood_batch([msa], target_index, with_masking, verbose, mask_entire_sequence, count_gaps, mask_distance)[0]

    def log_likelihood_batch(self, msa_list, target_index=-1, with_masking=True, verbose=False,
                       mask_entire_sequence=False, count_gaps=False, mask_distance=float("inf"), batch_size=None):
        """
            msa_list: a list of MSAs to calculate log_likelihood for.
            batch_size: number of MSAs to run on the gpu at once, if None, then batch_size=len(msa_list). default=None.

        """

        # Inspired by and borrowing code from:
        # https://github.com/facebookresearch/esm/blob/master/variant-prediction/predict.py

        if mask_entire_sequence and not with_masking:
            raise ValueError("you can't have mask_entire_sequence = True, and with_masking = False, it just doesn't make any sense!")

        gap_tokens = {self.model.alphabet.get_idx(x) for x in ESM_MSA_GAP_CHARACTERS}
        n_msas = len(msa_list)
        if batch_size is None:
            batch_size = n_msas
        log_likelihood_sum = [0.0 for _ in range(n_msas)]

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

        #TODO: setting mask_distance = 1 is equivalent to setting mask_entire_sequence, so we can probably get rid of mask_entire_sequence.
        with torch.no_grad():
            original_tokens = tokens[:, target_index].clone().detach()

            if (with_masking and mask_entire_sequence) or (not with_masking):
                if mask_entire_sequence:
                    for idx in range(range_start, overall_range_end):
                        tokens[:, target_index, idx] = self.model.alphabet.mask_idx
                
                for batch_start in range(0,tokens.shape[0], batch_size):

                    token_probs = torch.log_softmax(self.model.model(tokens[batch_start:batch_start+batch_size,:,:])['logits'], dim=-1)

                    for i_sample in range(token_probs.shape[0]):
                        for idx in range(range_start, msa_range_end[i_sample+batch_start]):
                            if count_gaps or original_tokens[i_sample+batch_start, idx].item() not in gap_tokens: # only add the likelihood to the running sum if we are counting gaps, or if the position does not contain a gap.
                                log_likelihood_sum[i_sample+batch_start] += token_probs[i_sample, target_index, idx, original_tokens[i_sample+batch_start, idx].item()]

                return [float(l_sum / msa_denominator[idx]) for idx, l_sum in enumerate(log_likelihood_sum)]

            elif with_masking:
                results = []
                for msa_idx in range(n_msas):
                    likelihood_sum = 0.0
                    original_msas = reformatted_msas[msa_idx] #is it possible to copy a slice from tokens?
                    original_string = original_msas[target_index][1]
                    all_samples_for_this_msa = []

                    num_samples_for_this_msa = min(mask_distance, len(original_string))
                    for _ in range(num_samples_for_this_msa): #I think you can do this with a pytorch/numpy broadcast of a slice from tokens
                        sample = original_msas.copy() 
                        sample[target_index] = (str(target_index), original_string) # this doesn't seem necessary?
                        all_samples_for_this_msa.append(sample.copy())

                    masked_idx = set()
                    _, _, all_samples_for_this_msa_tokens = self.model.batch_converter(all_samples_for_this_msa)
                    # all_samples_for_this_msa_tokens = all_samples_for_this_msa_tokens.cuda() if self.cuda else all_samples_for_this_msa_tokens
                    for i_sample in range(num_samples_for_this_msa):
                        for idx_pos in range(range_start, msa_range_end[msa_idx]):
                            # TODO this could be more efficient
                            if idx_pos % num_samples_for_this_msa == i_sample:
                                all_samples_for_this_msa_tokens[i_sample, target_index, idx_pos] = self.model.alphabet.mask_idx
                                masked_idx.add(idx_pos)
                    assert len(masked_idx) == msa_denominator[msa_idx], sorted(masked_idx)

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
                                # TODO this could be more efficient
                                if idx_pos % num_samples_for_this_msa == (i_sample + batch_start):
                                    if count_gaps or original_tokens[msa_idx, idx_pos].item() not in gap_tokens: # only add the likelihood to the running sum if we are counting gaps, or if the position does not contain a gap.
                                        likelihood_sum += token_probs[i_sample, target_index, idx_pos, original_tokens[msa_idx, idx_pos].item()]
                                        counted_idx.add(idx_pos)
                    assert len(counted_idx) == msa_denominator[msa_idx], sorted(counted_idx)
                        
                    
                    results.append(float(likelihood_sum / msa_denominator[msa_idx]))

                return results
