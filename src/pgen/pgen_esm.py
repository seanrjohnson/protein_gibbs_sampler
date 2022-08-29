import argparse
import textwrap
from pgen.esm_sampler import ESM_sampler
from pgen import models
from pgen.utils import write_sequential_fasta, RawAndDefaultsFormatter
from pathlib import Path
import sys


model_map = {"esm1b":models.ESM1b, "esm6":models.ESM6, "esm12":models.ESM12, "esm34":models.ESM34, "esm1v":models.ESM1v}

def main(input_h, output_p, args):

    sampler = ESM_sampler(model_map[args.model](),device=args.device)
    with open(output_p / "specification.tsv","w") as output_h:
        for line in input_h:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split("\t")
            if len(line) == 2:
                print("\t".join(line))
                print("\t".join(line), file=output_h)
                name = line[0]
                line_args = eval(line[1])
                sequences = sampler.generate(args.num_output_sequences, batch_size=args.batch_size, **line_args)
                write_sequential_fasta( output_p / (name + ".fasta"), sequences )
            else:
                print(f"Expected 2 values in specification file (name, line_args), got {len(line)}")
                print("\t".join(line))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Samples from an ESM BERT model to generate new protein sequences. 

            Input should be a tab separated file where columns are: 
            sample name, dict of sampler arguments

            """), 
            epilog=textwrap.dedent("""
            Available sampler arguments:

            seed_seq: protein msa to start from
            in_order: if True then cycle through the positions in order, otherwise randomly select positions each iteration.
            max_len: maximum size of each generated sequence. If None, then use the length of the longest input msa.
            leader_length: don't overwrite this many amino acids at the beginning of the sequence.
            leader_length_percent: if not None, then will set leader_length = int(len(seed_seq)*(leader_length_percent / 100))
            top_k: if >0, only sample from the top k most probable AAs
            temperature: higher numbers will mean there is a lower penalty for low-scoring amino acids.
            num_iters: how many times to run the forward loop for every batch. 
            burnin: during burn-in period, sample from full distribution; afterwards sample from top_k, set to 0 to never sample from full distribution (always take from top_k), or inf to always sample from full distribution.
            num_positions: generate new AAs for this many positions each iteration. If 0, then generate for all target positions each round.
            num_positions_percent: If not None, then set num_positions = int(len(seed_seq)*(num_positions_percent / 100))
            indexes: positions of the input sequence to modify. 1-indexed, if None then all positions after the leader.  
            
            """),
            formatter_class=RawAndDefaultsFormatter)
    parser.add_argument("-o", default=".", help="a directory to save the outputs to.")
    parser.add_argument("-i", default=None, help="tab separated file where the columns are as follows: [sample name] \\t [dict of arguments for the sampler].")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for sampling (sequences per iteration).")
    parser.add_argument("--num_output_sequences", type=int, default=1, help="total number of sequences to generate.")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, help="cpu or gpu")
    parser.add_argument("--model", type=str, default="esm1b", choices={"esm1b", "esm6", "esm12", "esm34", "esm1v"}, help="which model to use")

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