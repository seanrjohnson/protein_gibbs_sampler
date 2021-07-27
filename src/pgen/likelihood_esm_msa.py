import argparse
import textwrap
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen import models
from pgen.utils import parse_fasta, RawAndDefaultsFormatter, SequenceSubsetter
import sys
import tqdm

model_map = {"esm_msa1": models.ESM_MSA1}


def main(input_h, output_h, masking_off, device, model, mask_entire_sequence, target_index):
    sampler = ESM_MSA_sampler(model_map[model](), device=device)

    # TODO tqdm
    in_seqs = list(zip(*parse_fasta(input_h, return_names=True)))
    batch_msa = SequenceSubsetter.subset(in_seqs, len(in_seqs))  # TODO Do we want these to come from the file?
    score = sampler.log_likelihood(batch_msa, target_index=target_index, with_masking=not masking_off,
                                   mask_entire_sequence=mask_entire_sequence)
    print(f"{score}") #, file=output_h)
    output_h.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Calculates average log likelihood of a fasta ESM BERT model.
    
    writes a tab separated output file with columns:
    sequence name, score
    """),
                                     formatter_class=RawAndDefaultsFormatter)
    parser.add_argument("-o", type=str, default=None, help="")
    parser.add_argument("-i", default=None, help="A fasta file with sequences to calculate log likelihood for")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu", "gpu"}, help="cpu or gpu")
    parser.add_argument("--masking_off", action="store_true", default=False, help="If set, no masking is done.")
    parser.add_argument("--target_index", type=int, default=-1, help="the sequence in the msa to mask")
    parser.add_argument("--mask_entire_sequence", action="store_true", default=False,
                        help="If set, entire sequence is masked instead of a single value.")
    parser.add_argument("--model", type=str, default="esm_msa1", choices={"esm_msa1"},
                        help="which model to use")

    args = parser.parse_args()

    if args.i is not None:
        input_handle = open(args.i, "r")
    else:
        input_handle = sys.stdin

    if args.o is not None:
        output_handle = open(args.o, "w")
    else:
        output_handle = sys.stdout

    main(input_handle, output_handle, args.masking_off, args.device, args.model, args.mask_entire_sequence,
         int(args.target_index))

    if args.i is not None:
        input_handle.close()
    if args.o is not None:
        output_handle.close()
