import argparse, textwrap
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen import models
from pgen.utils import write_sequential_fasta, parse_fasta, SequenceSubsetter, RawAndDefaultsFormatter, run_phmmer, generate_alignment
from pathlib import Path
import sys
from tqdm import tqdm 
import math
import tempfile
import os
import warnings
from typing import List, Iterable

model_map = {"esm_msa1":models.ESM_MSA1}

def delete_msa_cols(msa:List[str], cols=Iterable):
    """
    Delete columns from a multiple sequence alignment. 
    msa: a list of strings, each string is a sequence in the alignment
    cols: an iterable of integers, the indices of the columns to delete

    returns: a new list of strings, each string is a sequence in the alignment with the columns deleted
    """
    cols = set(cols)
    return ["".join([seq[i] for i in range(len(seq)) if i not in cols]) for seq in msa]

def count_gaps_per_column(msa:List[str]):
    """
    Count the number of gaps in each column of a multiple sequence alignment
    msa: a list of strings, each string is a sequence in the alignment

    returns: a list of integers, the number of gaps in each column of the alignment
    """

    return [sum([seq[i] == "-" for seq in msa]) for i in range(len(msa[0]))]

def apply_gap_threshold(msa, gap_threshold):
    """
    Find columns in a multiple sequence alignment that exceed a certain gap percentage threshold
    msa: a list of strings, each string is a sequence in the alignment
    gap_threshold: a float, the percentage of sequences in the alignment that must contain a gap for the column to be considered for deletion

    returns: a list of integers, the indices of the columns that exceed the gap threshold
    """

    gap_counts = count_gaps_per_column(msa)
    gap_bound =  len(msa) * (gap_threshold / 100)
    return [i for i in range(len(gap_counts)) if gap_counts[i] > gap_bound]


def pgen_msa(templates_path, references_path, output_path, seqs_per_template, keep_identical, steps, passes, burn_in, device, model, alignment_size, ep, op, top_k, legacy=False, gap_percent_threshold=80, debug=False):
    # template_row = 0
    # if legacy:
    #     template_row = -1

    clean_flag = 'unalign'

    template_seqs = list(zip(*parse_fasta(templates_path, clean=clean_flag, return_names=True)))
    reference_seqs = parse_fasta(references_path, clean=clean_flag)

    gibbs_sampler = ESM_MSA_sampler(model_map[model](), device=device)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    write_sequential_fasta(tmp_file, reference_seqs)
    tmp_file.close()
    reference_db_path = tmp_file.name
    names, sequences = parse_fasta(reference_db_path,return_names=True)
    reference_seqs = dict()
    for i in range(len(names)):
        reference_seqs[names[i]] = sequences[i]
    del names
    del sequences

    with tqdm(total=len(template_seqs) * seqs_per_template) as pbar:
        with open(output_path,"w") as outfile:
            for template_name, template_seq in template_seqs:
                hits = run_phmmer(template_seq,reference_db_path,max_mode=debug)
                unaligned_seqs = list()
                unaligned_seqs.append(template_seq)
                
                for hit in hits:
                    if len(unaligned_seqs) == alignment_size:
                        break                    
                    if reference_seqs[hit] != template_seq or keep_identical:
                        unaligned_seqs.append(reference_seqs[hit])
                

                    
                if len(unaligned_seqs) < alignment_size:
                    warnings.warn(f"Warning: fewer than {alignment_size -1} hits found for template seq {template_name}")

                _, new_alignment = generate_alignment({"1": unaligned_seqs}, ep=ep, op=op) #mafft should preserve the order of sequences


                exclude_positions = []
                if not legacy: # not legacy mode, so delete gap positions in template seq
                    aligned_template_seq = new_alignment[0]
                    template_gaps = [i for i in range(len(aligned_template_seq)) if aligned_template_seq[i] == "-"]
                    new_alignment = delete_msa_cols(new_alignment, template_gaps)
                    exclude_positions = apply_gap_threshold(new_alignment, gap_percent_threshold)
                else: # legacy mode, so swap the first and last sequences to match the original implementation
                    tmp_seq = new_alignment[0]
                    new_alignment[0] = new_alignment[-1]
                    new_alignment[-1] = tmp_seq

                
                for i in range(seqs_per_template):
                    sampled_position = 0
                    if legacy:
                        sampled_position = -1
                    
                    new_seq = gibbs_sampler.generate_single(new_alignment, steps=steps, passes=passes, burn_in=burn_in, k=top_k, target_index=sampled_position, exclude_positions=exclude_positions)
                    new_seq = new_seq.replace("-","")
                    print(f">{i}_{template_name}\n{new_seq}", file=outfile, flush=True)
                    pbar.update(1)

    os.unlink(reference_db_path)

def main(argv):
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Samples from the ESM-MSA model to generate new protein sequences."""), 
            formatter_class=RawAndDefaultsFormatter)
    
    parser.add_argument("--templates", default=None, required=True, help="an unaligned fasta file with sequences to mask for generating new sequences.")
    parser.add_argument("--references", default=None, required=True, help="an unaligned fasta file with reference sequences to search for homologs to the templates.")
    parser.add_argument("-o", default=None, required=True, help="a fasta file to write generated sequences to")
    parser.add_argument("--seqs_per_template", type=int, default=1, help="Number of new sequences to generate for each template sequence.")
    parser.add_argument("--keep_identical", action="store_true", default=False, help="By default, if a template sequence is identical to the query sequence, it is thrown out. Set this if, for some reason you want to keep those.")
    parser.add_argument("--steps", type=int, default=10, help="Randomly assign the input positions to this many mask bins, and mask and generate over one bin at a time.")
    parser.add_argument("--passes", type=int, default=3, help="how many passes over the entire template sequence to make.")
    parser.add_argument("--burn_in", type=int, default=1, help="A number of passes equal to burn_in will sample from the entire distribution, after which amino acids will be sampled from the top_k most likely.")
    parser.add_argument("--top_k", type=int, default=1, help="Sample from the this many of the most probable amino acids, after burn in. If 0 then always sample from full distribution.")
    parser.add_argument("--legacy", action="store_true", default=False, help="Use the original implementation's behavior of swapping the first and last sequences in the alignment.")
    parser.add_argument("--gap_percent_threshold", type=float, default=80.0, 
                        help="Don't resample positions where more than this percent of sequences in the alignment contain gaps. Ignored in legacy mode."
                        " [Default: 80, meaning that if the MSA is 100 sequences, columns with 81 or more gaps will be not be resampled]")

    parser.add_argument("--ep", type=float, default=0.0, help="ep parameter passed to MAFFT for alignments")
    parser.add_argument("--op", type=float, default=1.53, help="op parameter passed to MAFFT for alignments")

    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, help="cpu or gpu") #TODO: allow specification of particular CUDA devices.
    parser.add_argument("--model", type=str, default="esm_msa1", choices={"esm_msa1"}, help="which model to use")
    parser.add_argument("--alignment_size", type=int, default=32, help="how many sequences (template plus references) should be in the alignments used for sequence generation.")
    parser.add_argument("--debug", action="store_true", default=False, help="run in debug mode. Runs phmmer in --max mode, to turn off pre-filters and allow finding very short hits.")

    args = parser.parse_args(argv)

    pgen_msa(args.templates, args.references, args.o, args.seqs_per_template, args.keep_identical, args.steps, args.passes, args.burn_in,
             args.device, args.model, args.alignment_size, args.ep, args.op, args.top_k, legacy=args.legacy, gap_percent_threshold=args.gap_percent_threshold,
             debug=args.debug
             )

if __name__ == "__main__":
    main(sys.argv[1:])
