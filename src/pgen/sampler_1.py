import torch
import math
import time
import random
from tqdm import trange

class Sampler_1():
    """adapted from bert-gen bert-babble.ipynb"""

    def __init__(self, model, device="cpu"):
        """
            model should be an object with parameters model, alphabet, and batch_converter
        """
        self.model = model

        #switch model to eval mode
        self.model.model.eval()
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
        out = [ "".join([self.model.alphabet.get_tok(seq[i]) for i in range(1,len(seq)) ]) for seq in batch]
   
        return out
        
    def generate_step(self, out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]
        
        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k 
            - return_list (Bool): if True, 
        """
        #TODO: repetition penalty.

        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample: #take from full distribution
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else: #take top hit, equivalent to top_k = 1
            idx = torch.argmax(logits, dim=-1) #TOOD: for batch size 1, this somehow returns data in a different format than the other ones.
        return idx.tolist() if return_list else idx

    def get_init_seq(self, seed_seq, max_len, batch_size = 1):
        """ Get initial sequence by padding seed_seq with masks """
        #In the paper they talk about padding with random sequence. I'm not sure that's a good idea. S.R.J.
        # Also, that code was commented out in the BertGen repo. So they probably didn't think was a good idea either.
        
        remaining_len = max_len - len(seed_seq)
        seed_seq = [x for x in seed_seq] #if input is a string, convert it to an array
        batch = [(str(i), seed_seq + ["<mask>"] * remaining_len) for i in range(batch_size)]
        
        #if rand_init:
        #    for ii in range(max_len):
        #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))
        labels, strs, tokens = self.model.batch_converter(batch)
        return tokens

    def generate(self, n_samples, seed_seq, batch_size=1, in_order=False, max_len=30, leader_length=0, top_k=0, temperature=None, num_iters=10,  burnin=float('inf'),
                            print_every_inner=10, print_every_outer=1, verbose=True, mask=True, num_positions=0, indexes=None, rollover_from_start=False):
        """ generate sequences

            n_samples: number of sequences to output
            seed_seq: protein sequence to start from
            batch_size: how many sequences to generate per loop.
            max_len: maximum size of each generated sequence
            sample: if >0, only sample from the top k most probable words
            top_k: if >0, only sample from the top k most probable AAs
            in_order: if True then cycle through the positions in order, otherwise randomly select positions each iteration
            leader_length: don't overwrite this many amino acids at the beginning of the sequence.
            temperature: 
            burnin: during burn-in period, sample from full distribution; afterwards take argmax, set to 0 to never sample (always take best), or inf to always sample
            num_iters: how many times to run the forward loop for every batch. 
            print_every: print after every this number of loops/batches
            num_positions: generate new AAs for this many positions each iteration. If 0, then generate for all target positions each round.
            indexes: positions of the input sequence to modify. 1-indexed, if None then all positions after the leader.

            #### Examples #####
            seed = "MTSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLLSHRPVRDIDRERDLLERLITLGKAHHLDAHYITRLFQLIIEDSVLTQQALLQQH"

            #To generate AAs one position at a time in order:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=True, num_positions=1, num_iters=len(seed), mask=True)
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

        cuda = self.cuda
        sequences = []
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()
        
        if leader_length < 0:
            leader_length = 0
        
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
                    else:
                        target_indexes = random.sample(indexes, num_positions)
                else:
                    target_indexes = indexes
                if mask:
                    for kk in target_indexes:
                        batch[:,kk] = self.model.alphabet.mask_idx

                out = self.model.model(batch)["logits"]
                for kk in target_indexes:
                    idxs = self.generate_step(out, gen_idx=kk, top_k=top_k, temperature=temperature, sample=(ii < burnin))
                    #print(idxs)
                    #if type(batch_size == 1: #TODO: probably better to handle this upstream in the squeeze step of generate_step
                    if type(idxs) == int:
                        # print(idxs)
                        # print()
                        batch[0][kk] = idxs
                    else:
                        for jj in range(batch_size):
                            batch[jj][kk] = idxs[jj]
                
            if batch_n == (n_batches - 1): #last batch, so maybe don't take all of them, just take enough to get to n_samples
                sequences += self.untokenize_batch(batch)[0:n_samples - len(sequences)]
            else:
                sequences += self.untokenize_batch(batch)
        return sequences

