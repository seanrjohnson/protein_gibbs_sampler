import argparse
import textwrap
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen.pgen_msa_revised import delete_msa_cols
from pgen import models
from pgen.utils import parse_fasta, RawAndDefaultsFormatter, SequenceSubsetter, add_to_msa, write_sequential_fasta, generate_alignment, run_phmmer
import sys
import tqdm
from tqdm.contrib import tzip
import tempfile
import os
import warnings
import re

# TODO: option to average across multiple runs with different random subsets
POSITIONAL_SCORE_SEP=";"


def main(
        input_h,
        output_h,
        masking_off,
        sampler,
        reference_msa_handle=None,
        in_msas=None, #Python dictionary where keys and values correspond to names and msas, respectively. If input msas are provided than reference_msa_handle will be ignored. The target or query sequence must be present at the very top of each input MSA
        delete_insertions=False,
        batch_size=1,
        subset_strategy="random",
        alignment_size=sys.maxsize,
        subset_random_seed=None,
        redraw=False,
        unaligned_queries=False,
#        count_gaps=False,
        mask_distance=float("inf"),
        csv=False,
        positionwise=None,
#        include_gaps_in_positionwise=False,
        keep_identical=False
    ):
    # output file(s) prep
    clean_flag = "delete" if delete_insertions else "upper"
    sep = "," if csv else "\t"

    print(f"id{sep}esm-msa", file=output_h)
    positionwise_h = None
    if positionwise is not None:
        positionwise_h = open(positionwise,"w")
        print(f"id{sep}esm-msa", file=positionwise_h)

    # MSA preprocessing
    seq_msa = None
    generate_msas = True
    if in_msas:
        generate_msas = False
    else:
        in_msas = {}
        reference_msa = parse_fasta(reference_msa_handle, clean=clean_flag)
        if subset_strategy == "top_hits": # if the strategy is top_hits, then we need to redraw and re-align for every query.
            tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
            write_sequential_fasta(tmp_file, reference_msa)
            tmp_file.close()
            reference_db_path = tmp_file.name
            renamed_reference_sequences = dict()
            for name, seq in zip(*parse_fasta(reference_db_path, return_names=True)):
                renamed_reference_sequences[name] = seq
        else:
            seq_msa = SequenceSubsetter.subset(reference_msa, alignment_size, strategy=subset_strategy, random_seed=subset_random_seed)

    # set in_seqs
    in_seqs = {}
    for name, seq in zip(*parse_fasta(input_h, return_names=True, clean=clean_flag)):
        in_seqs[name] = seq

    # get input sequences, and their names, and compute corresponding MSA
    def get_in_msa(name, seq_msa=seq_msa):
        seq = in_seqs[name]
        if generate_msas:
            # generate msa using top_hits from reference sequences
            if subset_strategy == "top_hits":
                hits = run_phmmer(seq, reference_db_path)
                # mafft should preserve the order of sequences

                unaligned_seqs = [seq]
                for hit in hits:
                    if len(unaligned_seqs) == alignment_size:
                        break                    
                    if renamed_reference_sequences[hit] != seq or keep_identical:
                        unaligned_seqs.append(renamed_reference_sequences[hit])
                if len(unaligned_seqs) < alignment_size:
                    warnings.warn(f"Warning: fewer than {alignment_size -1} hits found for template seq {name}")

                _, new_alignment = generate_alignment({"1": unaligned_seqs})
                return new_alignment
            else:
                if redraw:
                    seq_msa = SequenceSubsetter.subset(reference_msa, alignment_size, strategy=subset_strategy, random_seed=subset_random_seed)
                    if subset_random_seed is not None:
                        subset_random_seed += 1000000
                if unaligned_queries:
                    return add_to_msa(seq_msa, seq) #adds seq to top of MSA
                else:
                    return [seq] + seq_msa.copy()
        else:
            return in_msas[name]

    # for batching
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for batch in tqdm.tqdm(batch(list(in_seqs.keys()), batch_size), total=len(in_seqs)):
        tmp_name_list = batch
        tmp_msa_list = [get_in_msa(x) for x in batch]

        # TODO: There might be issues if batch_size > 1 and the MSAs are not all the same width
        
        # delete gap columns in reference sequence
        for i, msa in enumerate(tmp_msa_list):
            aligned_template_seq = msa[0]
            template_gaps = [i for i in range(len(aligned_template_seq)) if aligned_template_seq[i] == "-"]
            msa = delete_msa_cols(msa, template_gaps)
            tmp_msa_list[i] = msa
            #TODO: add a gap threshold, like in pgen_msa_revised. The gap threshold will exclude certain positions from being scored, for example if there are not very many sequences in the MSA with no gaps at that position.
            


        scores_iter = sampler.log_likelihood_batch(tmp_msa_list, with_masking=not masking_off, count_gaps=False, mask_distance=mask_distance, batch_size=batch_size)
        for j, (score, positional_scores) in enumerate(scores_iter):
            #query_seq = tmp_msa_list[j][0]
            print(f"{tmp_name_list[j]}{sep}{score}", file=output_h)
            if positionwise_h is not None:
                # if count_gaps and not include_gaps_in_positionwise:
                #     degapped_positional_scores = []
                #     for seq_idx, char in enumerate(query_seq):
                #         if char != '-':
                #             degapped_positional_scores.append(positional_scores[seq_idx])
                #     positional_scores = degapped_positional_scores
                print(f"{tmp_name_list[j]}{sep}{POSITIONAL_SCORE_SEP.join([str(round(x,3) ) for x in positional_scores])}", file=positionwise_h)
        output_h.flush()
        if positionwise_h is not None:
            positionwise_h.flush()
    
    if positionwise_h is not None:
        positionwise_h.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Calculates average log likelihood of a fasta from the ESM-MSA model.
    
    writes a tab separated output file with columns:
    sequence name, score
    """), formatter_class=RawAndDefaultsFormatter)
    
    parser.add_argument("-o", type=str, default=None, help="")
    parser.add_argument("-i", default=None, help="A fasta file with sequences to calculate log likelihood for")
    parser.add_argument("--reference_msa", default=None, required=True, help="A fasta file with an msa to use as a reference. If subset_strategy is top_hits, then this should be an unaligned fasta of reference sequences.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, gpu (cuda:0), or cuda:[int]")
    parser.add_argument("--masking_off", action="store_true", default=False, help="If set, no masking is done.")
    parser.add_argument("--delete_insertions", action='store_true', default=False, help="If set, then remove all lowercase and '.' characters from input sequences. Default: convert lower to upper and '.' to '-'.") #might want to have the option to keep "." in the msa and convert lower to upper (which would be consistent with the vocabulary, which has ".", but does not have lowercase characters.)
    parser.add_argument("--alignment_size", type=int, default=sys.maxsize, help="Sample this many sequences from the reference alignment before doing gibbs sampling, recommended values are 31-255. Default: the entire reference alignment.")
    
    parser.add_argument("--keep_identical", action="store_true", default=False, help="By default, for subset_strategy=top_hits, if a template sequence is identical to the query sequence, it is thrown out. Set this if, for some reason you want to keep those.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for sampling (msa instances per iteration).")
    parser.add_argument("--subset_strategy", default="random", choices={"random","in_order","top_hits"}, help="How to subset the reference alignment to get it to the desired size. random: draw randomly, in_order: take the sequences listed first in the reference alignment, top_hits: run phmmer for each query against the reference sequences and use a MAFFT MSA of the top hits as the reference.")
    parser.add_argument("--subset_random_seed", default=None, type=int, help="Seed to start the random batch subsetter at. The seed will increment by 1000000 after each draw.")
    parser.add_argument("--redraw", action='store_true', default=False, help="If subset_strategy is random, by default a single random draw will be used for all calculations. If redraw is set, then a new random draw of reference sequences will be done for each target sequence.")
    parser.add_argument("--unaligned_queries",  action='store_true', default=False, help="If the input sequences are unaligned or come from a different alignment than the reference msa, then use muscle profile to add each sequence to the reference alignment.")
#    parser.add_argument("--count_gaps",  action='store_true', default=False, help="If true then average the log likelihoods over the coding positions as well as the gap positions. By default, gap positions are not considered in the sums and averages.")
    parser.add_argument("--mask_distance",  type=int, default=None, help="If set, then multiple positions will be masked at a time, with (mask_distance - 1) non-masked positions between each masked position. This will make the likelihood calculations faster. Default: mask positions one at a time.")
    parser.add_argument("--csv", action='store_true', default=False, help="If set, then outputs will be a csv files.")
    parser.add_argument("--positionwise",  type=str, default=None, help="If set, then write positionwise log likelihoods will be written to this file. Two columns, id and esm-msa. Values in second column are a ';' separated list.")
#    parser.add_argument("--include_gaps_in_positionwise",  action='store_true', default=False, help="If set, then write positionwise log likelihoods will include gap positions, otherwise gap positions will be omitted.")


    args = parser.parse_args()

    if args.redraw and args.subset_strategy == 'in_order':
        raise ValueError(f"redraw is set, but subset_strategy is 'in_order', so all the draws will be the same. That's probably not what you're trying to do.")

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
    
    sampler = ESM_MSA_sampler(models.ESM_MSA1(), device=args.device)
    main(
        input_h=input_handle,
        output_h=output_handle,
        masking_off=args.masking_off,
        sampler=sampler,
        reference_msa_handle=reference_msa_handle,
        delete_insertions=args.delete_insertions,
        batch_size=args.batch_size,
        subset_strategy=args.subset_strategy,
        alignment_size=args.alignment_size,
        subset_random_seed=args.subset_random_seed,
        redraw=args.redraw,
        unaligned_queries=args.unaligned_queries,
        #count_gaps=args.count_gaps,
        mask_distance=mask_distance,
        csv=args.csv,
        positionwise=args.positionwise,
        # include_gaps_in_positionwise=args.include_gaps_in_positionwise,
        keep_identical=args.keep_identical
    )

    reference_msa_handle.close()
    if args.i is not None:
        input_handle.close()
    if args.o is not None:
        output_handle.close()