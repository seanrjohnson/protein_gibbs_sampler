import argparse
from pgen import sampler_1
from pgen import vanilla_esm1b
from pgen.utils import write_sequential_fasta
from pathlib import Path
import shutil

def main(input_h, output_p, args):
    device = args.device
    sampler = sampler_1.Sampler_1(vanilla_esm1b.ESM1b(),device=args.device)
    output_h = open(output_p / "specification.tsv","w")
    for line in input_h:
        line = line.strip().split("\t")
        if len(line) == 2:
            print("\t".join(line))
            print("\t".join(line), file=output_h)
            name = line[0]
            line_args = eval(line[1])
            sequences = sampler.generate(args.total_sequences, batch_size=args.batch_size, **line_args)
            write_sequential_fasta( output_p / (name + ".fasta"), sequences )
    output_h.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default=".", help="a directory to save the outputs in")
    parser.add_argument("-i", default=None, help="tab separated file where the first column is the sample name and second column is a dict of arguments for the sampler.")
    parser.add_argument("--batch_size", type=int, default=1, required=True, help="batch size for sampling (sequences per iteration).")
    parser.add_argument("--total_sequences", type=int, default=1, required=True, help="total number of sequences to generate.")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, required=True, help="")

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