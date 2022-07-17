#DEPRECEATED

import argparse, textwrap
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen import models
from pgen.utils import write_sequential_fasta, parse_fasta, SequenceSubsetter, RawAndDefaultsFormatter, run_phmmer, generate_alignment
from pathlib import Path
import sys
import math
import tempfile
import os
import warnings
import pandas as pd
import numpy as np

model_map = {"esm_msa1":models.ESM_MSA1}



def pgen_msa(msa, outpath, steps, device, model):
    
    clean_flag = 'upper'

    msa = parse_fasta(msa, clean=clean_flag)

    gibbs_sampler = ESM_MSA_sampler(model_map[model](), device=device)

    if steps == None:
        steps = len(msa[-1])

    probs, toks = gibbs_sampler.probs_single(msa, steps=steps, show_progress_bar=True)

    consensus = [toks[i] for i in np.argmax(probs,axis=0)]
    position=[i+1 for i in range(len(msa[-1]))]
    different = list()
    for i in range(len(consensus)):
        if consensus[i] == msa[-1][i]:
            different.append(0)
        else:
            different.append(1)
    

    table = pd.DataFrame(probs, columns=list(msa[-1]), index=toks)
    table = table.append(pd.DataFrame([position, consensus, different],index=["position", "consensus", "different"], columns=list(msa[-1])))
    table.to_csv(outpath, sep="\t")

    

def main(argv):
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Samples from the ESM-MSA model to generate new protein sequences."""), 
            formatter_class=RawAndDefaultsFormatter)
    
    parser.add_argument("--msa", default=None, required=True, help="calculate the probabilities for the last sequence in this MSA.")
    parser.add_argument("-o", default=None, required=True, help="a fasta file to write generated sequences to")
    parser.add_argument("--steps", type=int, default=None, help="Randomly assign the input positions to this many mask bins pass, and mask and generate over one bin at a time.")

    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, help="cpu or gpu")
    parser.add_argument("--model", type=str, default="esm_msa1", choices={"esm_msa1"}, help="which model to use")
    
    args = parser.parse_args(argv)

    pgen_msa(args.msa, args.o, args.steps, args.device, args.model)

if __name__ == "__main__":
    main(sys.argv[1:])
