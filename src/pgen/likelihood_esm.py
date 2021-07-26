import argparse
import textwrap
from pgen.esm_sampler import ESM_sampler
from pgen import models
from pgen.utils import parse_fasta, RawAndDefaultsFormatter
from pathlib import Path
import sys
import tqdm


model_map = {"esm1b":models.ESM1b, "esm6":models.ESM6, "esm12":models.ESM12, "esm34":models.ESM34, "esm1v":models.ESM1v}

def main(input_h, output_h, masking_off, device, model):

    sampler = ESM_sampler(model_map[model](),device=device)
    
    in_seqs = list(zip(*parse_fasta(input_h, return_names=True)))
    for name, seq in tqdm.tqdm(in_seqs):
        score = sampler.log_likelihood(seq, with_masking=not masking_off)
        print(f"{name}\t{score}",file=output_h)
        output_h.flush()
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Calculates average log likelihood of a fasta ESM BERT model.
    
    writes a tab separated output file with columns:
    sequence name, score
    """), 
            formatter_class=RawAndDefaultsFormatter)
    parser.add_argument("-o", type=str, default=None, help="")
    parser.add_argument("-i", default=None, help="A fasta file with sequences to calculate log likelihood for")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, help="cpu or gpu")
    parser.add_argument("--masking_off", action="store_true", default=False, help="If set, no masking is done.")
    parser.add_argument("--model", type=str, default="esm1v", choices={"esm1b", "esm6", "esm12", "esm34", "esm1v"}, help="which model to use")

    args = parser.parse_args()

    if args.i is not None:
        input_handle=open(args.i, "r")
    else:
        input_handle = sys.stdin

    if args.o is not None:
        output_handle=open(args.o, "w")
    else:
        output_handle = sys.stdout

    main(input_handle, output_handle, args.masking_off, args.device, args.model)

    if args.i is not None:
        input_handle.close()
    if args.o is not None:
        output_handle.close()
