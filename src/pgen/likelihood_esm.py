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

def main(input_h, output_h, masking_off, device, model, batch_size, mask_distance, csv, score_name):

    sampler = ESM_sampler(model_map[model](),device=device)
    
    in_seqs = list(zip(*parse_fasta(input_h, return_names=True, clean="unalign")))

    sep="\t"
    if csv:
        sep=","

    
    if score_name is None:
        score_name = model

    print(f"id{sep}{score_name}", file=output_h)
    tmp_seq_list = list()
    tmp_name_list = list()
    for i in tqdm.trange(len(in_seqs)):
        name, seq = in_seqs[i]
        tmp_seq_list.append(seq)
        tmp_name_list.append(name)
        if len(tmp_seq_list) == batch_size or i+1 == len(in_seqs):
            #TODO: batching is a little weird still because it used to be solely based on len(tmp_seq_list), but now batch size is independent of len(tmp_seq_list)
            scores = sampler.log_likelihood_batch(tmp_seq_list, with_masking=not masking_off, mask_distance=mask_distance,batch_size=batch_size)
            for j in range(len(scores)):
                print(f"{tmp_name_list[j]}{sep}{scores[j]}", file=output_h)
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
    parser.add_argument("--mask_distance",  type=int, default=None, help="If set, then multiple positions will be masked at a time, with (mask_distance - 1) non-masked positions between each masked position. This will make the likelihood calculations faster. Default: mask positions one at a time.")
    parser.add_argument("--model", type=str, default="esm1v", choices={"esm1b", "esm6", "esm12", "esm34", "esm1v"}, help="Which model to use.")
    parser.add_argument("--csv",  action='store_true', default=False, help="If set, then output will be a csv file.")
    parser.add_argument("--score_name",  type=str, default=None, help="For csv output, what to put as the second column name.")

    args = parser.parse_args()

    if args.i is not None:
        input_handle=open(args.i, "r")
    else:
        input_handle = sys.stdin

    if args.o is not None:
        output_handle=open(args.o, "w")
    else:
        output_handle = sys.stdout

    mask_distance = float("inf")
    if args.mask_distance is not None:
        mask_distance = args.mask_distance
    if mask_distance < 1:
        raise ValueError(f"mask distance must be an integer >= 1.")

    if args.masking_off and args.mask_distance is not None:
        raise ValueError(f"--masking_off and --mask_distance are both set, that doesn't make sense.")

    main(input_handle, output_handle, args.masking_off, args.device, args.model, args.batch_size, mask_distance, args.csv, args.score_name)

    if args.i is not None:
        input_handle.close()
    if args.o is not None:
        output_handle.close()
