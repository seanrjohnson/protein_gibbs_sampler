import argparse, textwrap
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen import models
from pgen.utils import write_sequential_fasta, parse_fasta, SequenceSubsetter, RawAndDefaultsFormatter
from pathlib import Path
import sys
from tqdm import trange
import math

model_map = {"esm_msa1":models.ESM_MSA1}


def main(input_h, output_p, args):
    clean_flag = 'upper'
    if args.delete_insertions:
        clean_flag = 'delete'

    gibbs_sampler = ESM_MSA_sampler(model_map[args.model](), device=args.device)

    

    with open(output_p / "specification.tsv","w") as output_h:
        
        for line in input_h:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split("\t")
            if len(line) == 3:
                print("\t".join(line))
                print("\t".join(line), file=output_h)
                name = line[0]
                line_args = eval(line[1])

                input_msa = parse_fasta(line[2], clean=clean_flag)
                alignment_size = args.alignment_size
                if alignment_size == sys.maxsize:
                    alignment_size = len(input_msa)
                
                batches = math.ceil(args.num_output_sequences / alignment_size )
                sequences = list()
                for i in trange(batches):
                    batch_msa = SequenceSubsetter.subset(input_msa, alignment_size, args.keep_first_sequence, args.subset_strategy)
                    sequences += gibbs_sampler.generate(n_samples=len(batch_msa), seed_msa=batch_msa, batch_size=args.batch_size, show_progress_bar=False, **line_args)
                write_sequential_fasta( output_p / (name + ".fasta"), sequences[0:args.num_output_sequences] )
            else:
                print(f"Expected 3 values in specification file (name, line_args, input_msa), got {len(line)}")
                print("\t".join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent("""Samples from the ESM-MSA model to generate new protein sequences. 

            Input should be a tab separated file where columns are: 
            sample name, dict of sampler arguments, fasta of seed sequences

            """), 
            epilog=textwrap.dedent("""
            Available sampler arguments:

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

    parser.add_argument("-o", default=".", help="a directory to save the outputs in")
    parser.add_argument("-i", default=None, help="tab separated file where the columns are as follows: [sample name] \\t [dict of arguments for the sampler] \\t [path to seed msa in fasta or a2m format].")
    parser.add_argument("--batch_size", type=int, default=1, choices={1,}, help="batch size for sampling (msa instances per iteration). Must be 1. This might change in the future.")
    parser.add_argument("--num_output_sequences", type=int, default=1, help="total number of sequences to generate.")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, help="cpu or gpu")
    parser.add_argument("--model", type=str, default="esm_msa1", choices={"esm_msa1"}, help="which model to use")
    parser.add_argument("--delete_insertions", action='store_true', default=False, help="If set, then remove all lowercase and '.' characters from input sequences. Default: convert lower to upper and '.' to '-'.") #might want to have the option to keep "." in the msa and convert lower to upper (which would be consistent with the vocabulary, which has ".", but does not have lowercase characters.)

    parser.add_argument("--alignment_size", type=int, default=sys.maxsize, help="Sample this many sequences from the input alignment before doing gibbs sampling, recommended values are 32-256. Default: the entire input alignment.")

    parser.add_argument("--keep_first_sequence", action='store_true', default=False, help="If set, then keep the first sequence and sample the rest according to subset_strategy.")
    parser.add_argument("--subset_strategy", default="random", choices=SequenceSubsetter.subset_strategies, help="How to subset the input alignment to get it to the desired size.")

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