import argparse
import textwrap
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen import models
from pgen.utils import parse_fasta, RawAndDefaultsFormatter, SequenceSubsetter
import sys
import tqdm

model_map = {"esm_msa1": models.ESM_MSA1}

# TODO: option to average across multiple

def main(input_h, output_h, masking_off, device, model, mask_entire_sequence, reference_msa_path, delete_insertions, batch_size, subset_strategy, alignment_size):
    clean_flag = 'upper'
    if delete_insertions:
        clean_flag = 'delete'

    sampler = ESM_MSA_sampler(model_map[model](), device=device)

    in_seqs = list(zip(*parse_fasta(input_h, return_names=True, clean=clean_flag)))
    reference_msa = parse_fasta(reference_msa_path, clean=clean_flag)

    tmp_seq_list = list()
    tmp_name_list = list()
    for i in tqdm.trange(len(in_seqs)):
        name, seq = in_seqs[i]
        tmp_seq_list.append(seq)
        tmp_name_list.append(name)
        if len(tmp_seq_list) == batch_size or i+1 == len(in_seqs):
            batch_msa = SequenceSubsetter.subset(reference_msa, alignment_size, strategy=subset_strategy)
            
            scores = sampler.log_likelihood_batch([batch_msa.copy() + [s] for s in tmp_seq_list], with_masking=not masking_off, mask_entire_sequence=mask_entire_sequence)
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
    parser.add_argument("-i", default=None, help="A fasta file with sequences to calculate log likelihood for")
    parser.add_argument("--reference_msa", default=None, required=True, help="A fasta file with an msa to use as a reference")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu", "gpu"}, help="cpu or gpu")
    parser.add_argument("--masking_off", action="store_true", default=False, help="If set, no masking is done.")
    parser.add_argument("--mask_entire_sequence", action="store_true", default=False,
                        help="If set, entire sequence is masked instead of a single value.")
    parser.add_argument("--delete_insertions", action='store_true', default=False, help="If set, then remove all lowercase and '.' characters from input sequences. Default: convert lower to upper and '.' to '-'.") #might want to have the option to keep "." in the msa and convert lower to upper (which would be consistent with the vocabulary, which has ".", but does not have lowercase characters.)
    parser.add_argument("--alignment_size", type=int, default=sys.maxsize, help="Sample this many sequences from the reference alignment before doing gibbs sampling, recommended values are 31-255. Default: the entire reference alignment.")
    
    parser.add_argument("--model", type=str, default="esm_msa1", choices={"esm_msa1"},
                        help="which model to use.")
    parser.add_argument("--batch_size", type=int, default=1, choices={1,}, help="Batch size for sampling (msa instances per iteration).")
    parser.add_argument("--subset_strategy", default="random", choices=SequenceSubsetter.subset_strategies, help="How to subset the reference alignment to get it to the desired size.")

    args = parser.parse_args()

    if args.i is not None:
        input_handle = open(args.i, "r")
    else:
        input_handle = sys.stdin

    if args.o is not None:
        output_handle = open(args.o, "w")
    else:
        output_handle = sys.stdout

    main(input_handle, output_handle, args.masking_off, args.device, args.model, args.mask_entire_sequence, args.reference_msa, args.delete_insertions, args.batch_size, args.subset_strategy, args.alignment_size)

    if args.i is not None:
        input_handle.close()
    if args.o is not None:
        output_handle.close()
