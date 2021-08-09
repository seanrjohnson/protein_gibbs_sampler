from pgen.utils import parse_fasta
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default=None)
    parser.add_argument("-o", default=None)
    parser.add_argument("--clean_strategy", type=str, default=None, choices={"delete", "upper", "unalign"}, required=True, help="")
    parser.add_argument('--full_name', action='store_true', default=False, help="if true then keep the whole name of the sequences, including the description")

    args = parser.parse_args()

    if args.i is not None:
        input_handle=open(args.i, "r")
    else:
        input_handle = sys.stdin

    if args.o is not None:
        output_handle = open(args.o, "w")
    else:
        output_handle = sys.stdout


    names, seqs = parse_fasta(input_handle, return_names=True, clean=args.clean_strategy, full_name=args.full_name)

    for i in range(len(names)):
        print(f">{names[i]}\n{seqs[i]}", file=output_handle)

    if args.i is not None:
        input_handle.close()

    if args.o is not None:
        output_handle.close()