import argparse
from pgen.esm_sampler import ESM_sampler
from pgen import models
from pgen.utils import write_sequential_fasta
from pathlib import Path
import shutil

model_map = {"esm1b":models.ESM1b, "esm6":models.ESM6, "esm12":models.ESM12, "esm34":models.ESM34}

def main(input_h, output_p, args):
    device = args.device
    sampler = ESM_sampler(model_map[args.model](),device=args.device)
    output_h = open(output_p / "specification.tsv","w")
    for line in input_h:
        line = line.strip().split("\t")
        if len(line) == 3:
            print("\t".join(line))
            print("\t".join(line), file=output_h)
            name = line[0]
            line_args = eval(line[1])
            sequences = sampler.generate(args.total_sequences, seed_seq=line[2].upper(), batch_size=args.batch_size, **line_args)
            write_sequential_fasta( output_p / (name + ".fasta"), sequences )
    output_h.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default=".", help="a directory to save the outputs in")
    parser.add_argument("-i", default=None, help="tab separated file where the columns are as follows: [sample name] \\t [dict of arguments for the sampler] \\t [path to seed msa in fasta format].")
    parser.add_argument("--batch_size", type=int, default=1, required=True, help="batch size for sampling (sequences per iteration).")
    parser.add_argument("--total_sequences", type=int, default=1, required=True, help="total number of sequences to generate.")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, required=True, help="cpu or gpu")
    parser.add_argument("--model", type=str, default="esm1b", choices={"esm1b", "esm6", "esm12", "esm34"}, help="which model to use")

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