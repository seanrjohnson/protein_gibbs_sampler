import argparse
import textwrap
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen import models
from pgen.utils import parse_fasta, RawAndDefaultsFormatter, SequenceSubsetter, add_to_msa
import sys
import tqdm



model_map = {"esm_msa1": models.ESM_MSA1}

# TODO: option to average across multiple runs with different random subsets




def main(input_h, output_h, masking_off, sampler, reference_msa_handle, delete_insertions, batch_size, subset_strategy, alignment_size, subset_random_seed=None, redraw=False, unaligned_queries=False, count_gaps=False, mask_distance=float("inf")):
    clean_flag = 'upper'
    if delete_insertions:
        clean_flag = 'delete'

    in_seqs = list(zip(*parse_fasta(input_h, return_names=True, clean=clean_flag)))
    reference_msa = parse_fasta(reference_msa_handle, clean=clean_flag)

    # tmp_seq_list = list()
    tmp_name_list = list()
    tmp_msa_list = list()

    seq_msa = SequenceSubsetter.subset(reference_msa, alignment_size, strategy=subset_strategy, random_seed=subset_random_seed)


    for i in tqdm.trange(len(in_seqs)):
        name, seq = in_seqs[i]
        tmp_name_list.append(name)

        if redraw:
            seq_msa = SequenceSubsetter.subset(reference_msa, alignment_size, strategy=subset_strategy, random_seed=subset_random_seed)
            if subset_random_seed is not None:
                subset_random_seed += 1000000

        if unaligned_queries:
            tmp_msa_list.append(add_to_msa(seq_msa, seq))
        else:     
            tmp_msa_list.append(seq_msa.copy() + [seq])            

        if len(tmp_msa_list) == batch_size or i+1 == len(in_seqs):
            #TODO: batching is a little weird still because it used to be solely based on len(tmp_msa_list), but now batch size is independent of len(tmp_msa_list)
            scores = sampler.log_likelihood_batch(tmp_msa_list, with_masking=not masking_off, count_gaps=count_gaps, mask_distance=mask_distance, batch_size=batch_size)
            for j in range(len(scores)):
                print(f"{tmp_name_list[j]}\t{scores[j]}", file=output_h)
            output_h.flush()
            tmp_msa_list = list()
            tmp_name_list = list()
        
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Calculates average log likelihood of a fasta from the ESM-MSA model.
    
    writes a tab separated output file with columns:
    sequence name, score
    """),
                                     formatter_class=RawAndDefaultsFormatter)
    parser.add_argument("-o", type=str, default=None, help="")
    parser.add_argument("-i", default=None, help="A fasta file with sequences to calculate log likelihood for")
    parser.add_argument("--reference_msa", default=None, required=True, help="A fasta file with an msa to use as a reference")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu", "gpu"}, help="cpu or gpu")
    parser.add_argument("--masking_off", action="store_true", default=False, help="If set, no masking is done.")
    parser.add_argument("--delete_insertions", action='store_true', default=False, help="If set, then remove all lowercase and '.' characters from input sequences. Default: convert lower to upper and '.' to '-'.") #might want to have the option to keep "." in the msa and convert lower to upper (which would be consistent with the vocabulary, which has ".", but does not have lowercase characters.)
    parser.add_argument("--alignment_size", type=int, default=sys.maxsize, help="Sample this many sequences from the reference alignment before doing gibbs sampling, recommended values are 31-255. Default: the entire reference alignment.")
    
    parser.add_argument("--model", type=str, default="esm_msa1", choices={"esm_msa1"},
                        help="which model to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for sampling (msa instances per iteration).")
    parser.add_argument("--subset_strategy", default="random", choices=SequenceSubsetter.subset_strategies, help="How to subset the reference alignment to get it to the desired size.")
    parser.add_argument("--subset_random_seed", default=None, type=int, help="Seed to start the random batch subsetter at. The seed will increment by 1000000 after each draw.")
    parser.add_argument("--redraw", action='store_true', default=False, help="If subset_strategy is random, by default a single random draw will be used for all calculations. If redraw is set, then a new random draw of reference sequences will be done for each target sequence.")
    parser.add_argument("--unaligned_queries",  action='store_true', default=False, help="If the input sequences are unaligned or come from a different alignment than the reference msa, then use muscle profile to add each sequence to the reference alignment.")
    parser.add_argument("--count_gaps",  action='store_true', default=False, help="If true then average the log likelihoods over the coding positions as well as the gap positions. By default, gap positions are not considered in the sums and averages.")
    parser.add_argument("--mask_distance",  type=int, default=None, help="If set, then multiple positions will be masked at a time, with (mask_distance - 1) non-masked positions between each masked position. This will make the likelihood calculations faster. Default: mask positions one at a time.")


    args = parser.parse_args()

    if args.redraw and args.subset_strategy != 'random':
        raise ValueError(f"redraw is set, but subset_strategy is not 'random', so all the draws will be the same. That's probably not what you're trying to do.")

    mask_distance = float("inf")
    if args.mask_distance is not None:
        mask_distance = args.mask_distance

    if mask_distance < 1:
        raise ValueError(f"mask distance must be an integer >= 1.")



    if args.i is not None:
        input_handle = open(args.i, "r")
    else:
        input_handle = sys.stdin

    if args.o is not None:
        output_handle = open(args.o, "w")
    else:
        output_handle = sys.stdout

    reference_msa_handle = open(args.reference_msa, "r")
    
    sampler = ESM_MSA_sampler(model_map[args.model](), device=args.device)
    main(input_handle, output_handle, args.masking_off, sampler, reference_msa_handle, args.delete_insertions, args.batch_size, args.subset_strategy, args.alignment_size, args.subset_random_seed, args.redraw, args.unaligned_queries, args.count_gaps, mask_distance)

    reference_msa_handle.close()
    if args.i is not None:
        input_handle.close()
    if args.o is not None:
        output_handle.close()
