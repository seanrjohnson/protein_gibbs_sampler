import torch
import math
import time
import random
from tqdm import trange
from pgen.esm_sampler import generate_step


class ESM_MSA_sampler():
    """adapted from bert-gen bert-babble.ipynb"""
    
    def __init__(self, model, device="cpu"):
        """
            model should be an object with parameters model, alphabet, and batch_converter
        """
        
        #CONVERTED TO MSA  (but not tested)

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

    def untokenize_batch(self, batch): #TODO: maybe should be moved to the model class, or a model superclass?
        #convert tokens to AAs, but skip the first one, because that one is <cls>
        
        #CONVERTED TO MSA  (but not tested)

        out_batch = list()
        for batch_index in range(len(batch)):
            msa = batch[batch_index]
            out_msa = [ "".join([self.model.alphabet.get_tok(seq[i]) for i in range(1,len(seq)) ]) for seq in msa]
            out_batch.append(out_msa)

        return out


    def get_init_msa(self, seed_msa, max_len, batch_size = 1):
        """ Get initial msa by padding seed_seq with masks, and then tokenizing."""
        
        #CONVERTED TO MSA (but not tested)

        padded_msa = list()
        for i, seq in enumerate(seed_msa):
            remaining_len = max_len - len(seq)
            seq = [x for x in seq] #if input is a string, convert it to an array
            padded_msa.append( (str(i), seq + ["<mask>"] * remaining_len) )
        
        labels, strs, tokens = self.model.batch_converter([padded_msa] * batch_size)
        return tokens

    def generate(self, n_samples, seed_msa, batch_size=1, in_order=False, max_len=None, leader_length=0, leader_length_percent=None, top_k=0, temperature=None, num_iters=10,  burnin=float('inf'),
                            mask=True, num_positions=0, num_positions_percent=None, indexes=None, rollover_from_start=False):
        """ generate sequences

            n_samples: number of sequences to output
            seed_seq: protein msa to start from
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


        #TODO: CONVERT THIS TO MSA!

        #TODO: repetition penalty, somehow?
        #TODO: add dilated sequential sampling, like sampling every third or fifth amino acid and then doing the whole protein in like 3 or 5 steps, or something like that.
        with torch.no_grad(): # I'm not sure if this no_grad is necessary or not, but it couldn't hurt!

            cuda = self.cuda
            sequences = []
            n_batches = math.ceil(n_samples / batch_size)
            start_time = time.time()
            
            if num_positions_percent is not None:
                num_positions = int(len(seed_seq)*(num_positions_percent / 100))
            if num_positions < 0:
                num_positions = 0

            if leader_length_percent is not None:
                leader_length = int(len(seed_seq)*(leader_length_percent / 100))
            if leader_length < 0:
                leader_length = 0
            
            if max_len is None:
                max_len = len(seed_seq)

            for batch_n in trange(n_batches):

                batch = self.get_init_seq(seed_seq, max_len, batch_size)
                batch = batch.cuda() if cuda else batch

                if indexes is None:
                    indexes = range(1,max_len+1) #skip position 1, because that should be <cls>
                    if rollover_from_start == False: #we rollover from the end of the leader sequence
                        indexes = [i for i in indexes if i > leader_length]
                        last_i = leader_length - 1
                    else:
                        last_i = -1
                else:
                    last_i = -1
                
                if num_positions > len(indexes):
                    num_positions = len(indexes)

                for ii in range(num_iters):
                    if num_positions > 0: #do some subset of positions
                        if in_order: #cycle through the indexes
                            next_i = last_i 
                            sampled = 0
                            target_indexes = list()
                            while sampled < num_positions:
                                sampled += 1
                                next_i = (next_i+1) % len(indexes)
                                target_indexes.append(indexes[next_i])
                            last_i = next_i
                            target_indexes = [target_indexes] * batch_size
                        else:
                            target_indexes = list()
                            for b in range(batch_size):
                                target_indexes.append(random.sample(indexes, num_positions))
                    else:
                        target_indexes = [indexes] * batch_size
                    
                    if mask:
                        for batch_index in range(len(target_indexes)):
                            for kk in target_indexes[batch_index]:
                                batch[b,kk] = self.model.alphabet.mask_idx

                    out = self.model.model(batch)["logits"]
                    
                    for batch_index in range(batch_size):
                        for kk in target_indexes[batch_index]:
                            idx = generate_step(out[batch_index], gen_idx=kk, top_k=top_k, temperature=temperature, sample=(ii < burnin))
                            batch[batch_index][kk] = idx
                if batch_n == (n_batches - 1): #last batch, so maybe don't take all of them, just take enough to get to n_samples
                    sequences += self.untokenize_batch(batch)[0:n_samples - len(sequences)]
                else:
                    sequences += self.untokenize_batch(batch)
            return sequences

