import argparse
import textwrap
from pgen.esm_sampler import ESM_sampler
from pgen import models
from pgen.utils import parse_fasta, RawAndDefaultsFormatter, unalign
from pathlib import Path
import sys
import tqdm
import math


model_map = {"esm1b":models.ESM1b, "esm6":models.ESM6, "esm12":models.ESM12, "esm34":models.ESM34, "esm1v":models.ESM1v}

def main(input_h, output_h, masking_off, device, model, batch_size):

    sampler = ESM_sampler(model_map[model](),device=device)
    
    in_seqs = list(zip(*parse_fasta(input_h, return_names=True, clean="unalign")))

    tmp_seq_list = list()
    tmp_name_list = list()
    for i in tqdm.trange(len(in_seqs)):
        name, seq = in_seqs[i]
        tmp_seq_list.append(seq)
        tmp_name_list.append(name)
        if len(tmp_seq_list) == batch_size or i+1 == len(in_seqs):
            scores = sampler.log_likelihood_batch(tmp_seq_list, with_masking=not masking_off)
            for j in range(len(scores)):
                print(f"{tmp_name_list[j]}\t{scores[j]}", file=output_h)
            output_h.flush()
            tmp_seq_list = list()
            tmp_name_list = list()

        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Calculates average log likelihood of a fasta ESM BERT model.
    
    writes a tab separated output file with columns:
    sequence name, score
    """), 
            formatter_class=RawAndDefaultsFormatter)
    parser.add_argument("-o", type=str, default=None, help="")
    parser.add_argument("-i", default=None, help="A fasta file with sequences to calculate log likelihood for. Any gaps or stop codons will be removed before running the ")
    parser.add_argument("--batch_size", default=1, help="How many sequences to batch together.")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, help="cpu or gpu")
    parser.add_argument("--masking_off", action="store_true", default=False, help="If set, no masking is done.")
    parser.add_argument("--model", type=str, default="esm1v", choices={"esm1b", "esm6", "esm12", "esm34", "esm1v"}, help="Which model to use.")

    args = parser.parse_args()

    if args.i is not None:
        input_handle=open(args.i, "r")
    else:
        input_handle = sys.stdin

    if args.o is not None:
        output_handle=open(args.o, "w")
    else:
        output_handle = sys.stdout

    main(input_handle, output_handle, args.masking_off, args.device, args.model, args.batch_size)

    if args.i is not None:
        input_handle.close()
    if args.o is not None:
        output_handle.close()
