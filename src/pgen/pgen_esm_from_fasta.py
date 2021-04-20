import sys
import argparse
from pgen.esm_sampler import ESM_sampler
from pgen import models
from pgen.utils import write_sequential_fasta, parse_fasta, unalign, add_gaps_back
from pathlib import Path
import random
from tqdm import trange

model_map = {"esm1b":models.ESM1b, "esm6":models.ESM6, "esm12":models.ESM12, "esm34":models.ESM34}

def main(input_h, output_p, args):
    sampler = ESM_sampler(model_map[args.model](),device=args.device)
    with open(output_p / "specification.tsv","w") as output_h:
        for line in input_h:
            line = line.strip().split("\t")
            if len(line) == 3:
                print("\t".join(line))
                print("\t".join(line), file=output_h)
                name = line[0]
                line_args = eval(line[1])
                seeds = parse_fasta(line[2], clean=None)
                sequences = list()
                for out_seq_i in trange(args.num_output_sequences):
                    seed = random.choice(seeds)
                    seed, gap_mask = unalign(seed)
                    generated_sequence = sampler.generate(n_samples=1, seed_seq=seed, batch_size=args.batch_size, **line_args)[0]
                    if args.keep_gap_positions:
                        generated_sequence = add_gaps_back(generated_sequence, gap_mask)
                    sequences += [generated_sequence]
                write_sequential_fasta( output_p / (name + ".fasta"), sequences )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default=".", help="a directory to save the outputs in")
    parser.add_argument("-i", default=None, help="tab separated file where the columns are as follows: [sample name] \\t [dict of arguments for the sampler] \\t [seed sequence].")
    parser.add_argument("--batch_size", type=int, default=1, choices={1,}, help="batch size for sampling (sequences per iteration). Must be 1. This might change in the future.")
    parser.add_argument("--num_output_sequences", type=int, default=1, help="total number of sequences to generate.")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, help="cpu or gpu")
    parser.add_argument("--model", type=str, default="esm1b", choices={"esm1b", "esm6", "esm12", "esm34"}, help="which model to use")
    parser.add_argument("--keep_gap_positions", action='store_true', default=False, help="If set, then the sampler will remember where the gaps are in the original sequence and add gaps back into the same positions after generating new sequences.")

    args = parser.parse_args()

    if args.i is not None:
        input_handle=open(args.i, "r")
    else:
        input_handle = sys.stdin

    output_path = Path(args.o)

    output_path.mkdir(exist_ok=True)

    main(input_handle, output_path, args)

    if args.i is not None:
        input_handle.close()