import os
from typing import Dict,Tuple
import io
# import pandas as pd
from pathlib import Path
import uuid
import subprocess
import numpy as np
import string
import random
import argparse
import tempfile
import os
import sys
from Bio import SearchIO


class RawAndDefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def _open_if_is_name(filename_or_handle, mode="r"):
    """
        if a file handle is passed, return the file handle
        if a Path object or path string is passed, open and return a file handle to the file.

        returns:
            file_handle, input_type ("name" | "handle")
    """
    out = filename_or_handle
    input_type = "handle"
    try:
        out = open(filename_or_handle, mode)
        input_type = "name"
    except TypeError:
        pass
    except Exception as e:
        raise(e)

    return (out, input_type)


def unalign(sequence: str) -> Tuple[str,list]:
    """
        input:
            sequence: the starting sequence
                if 'unalign' then convert to upper, delete ".", "*", "-"

        output:
            cleaned_sequence: the cleaned sequence
            gap_mask: a list containing chars or None. The idea is that to get a sequence with gaps in the same places
    """
    upperified = sequence.upper()
    acceptable = string.ascii_uppercase
    cleaned_list = list()
    gap_mask = list()
    for c in upperified:
        if c in string.ascii_uppercase:
            cleaned_list.append(c)
            gap_mask.append(None)
        else:
            gap_mask.append(c)
    return "".join(cleaned_list), gap_mask

def add_gaps_back(sequence: str, gap_mask: list) -> str:
    """
        input:
            sequence: the cleaned sequence
            gap_mask: a list containing chars or None. The idea is that to get a sequence with gaps in the same places, you will pull a character from the sequence for every "None" in the mask, and otherwise pull the mask character.
        output:
            a string of size len(gap_mask) where None positions have been replaced, in order, by characters from sequence.

        example:
            add_gaps_back("MTGQ", [None,'-','-',None,None,".","-",None,"*"])
                = "M--TG.-Q*"
    """
    out = list()
    seq_index = 0
    for c in gap_mask:
        if c is None:
            out.append(sequence[seq_index])
            seq_index += 1
        else:
            out.append(c)
    return "".join(out)


def parse_fasta(filename, return_names=False, clean=None, full_name=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        

        input:
            filename: the name of a fasta file or a filehandle to a fasta file.
            return_names: if True then return two lists: (names, sequences), otherwise just return list of sequences
            clean: {None, 'upper', 'delete', 'unalign'}
                    if 'delete' then delete all lowercase "." and "*" characters. This is usually if the input is an a2m file and you don't want to preserve the original length.
                    if 'upper' then delete "*" characters, convert lowercase to upper case, and "." to "-"
                    if 'unalign' then convert to upper, delete ".", "*", "-"
            full_name: if True, then returns the entire name. By default only the part before the first whitespace is returned.

        output: sequences or (names, sequences)
    """
    
    prev_len = 0
    prev_name = None
    prev_seq = ""
    out_seqs = list()
    out_names = list()
    (input_handle, input_type) = _open_if_is_name(filename)

    for line in input_handle:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == ">":
            if full_name:
                name = line[1:]
            else:
                parts = line.split(None, 1)
                name = parts[0][1:]
            out_names.append(name)
            if (prev_name is not None):
                out_seqs.append(prev_seq)
            prev_len = 0
            prev_name = name
            prev_seq = ""
        else:
            prev_len += len(line)
            prev_seq += line
    if (prev_name != None):
        out_seqs.append(prev_seq)

    if input_type == "name":
        input_handle.close()
    
    if clean == 'delete':
        # uses code from: https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)

        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i])
    
    elif clean == 'upper':
        deletekeys = {'*': None, ".": "-"}
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)

        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i].upper())
    elif clean == 'unalign':
        deletekeys = {'*': None, ".": None, "-": None}
        
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)
        
        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i].upper())
    elif clean is not None:
        raise ValueError(f"unrecognized input for clean parameter: {clean}")

    if return_names:
        return out_names, out_seqs
    else:
        return out_seqs


def add_to_msa(msa, new_seq):
    """
        calls muscle to append a new sequence to the end of an msa.

        Input:
            msa: a list of protein sequence strings, each of the same length.

            new_seq: a protein sequence string.

        output:
            out_msa: a list of protein sequence strings. The new sequence will be at the end of the list.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out1_name = os.path.join(tmp, 'out1.fasta')
        out2_name = os.path.join(tmp, 'out2.fasta')
        
        new_seq_name = "new_seq"

        write_sequential_fasta(out1_name, msa)
        with open(out2_name,"w") as out2:
            print(f">{new_seq_name}\n{new_seq}", file=out2)
        
        muscle_results = subprocess.run(["muscle", "-profile", "-in1", out1_name, "-in2", out2_name], capture_output=True, encoding="utf-8")
        names, seqs = parse_fasta_string(muscle_results.stdout, True)
        #https://www.geeksforgeeks.org/python-move-element-to-end-of-the-list/
        try:
            seqs.append(seqs.pop(names.index(new_seq_name)))
        except ValueError as error:
            print(error, file=sys.stderr)
            print(f"{names}", file=sys.stderr)
            print(muscle_results.stdout, file=sys.stderr)
            print(muscle_results.stderr, file=sys.stderr)
            raise error

    return seqs


def parse_fasta_string(fasta_string, return_names=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
        input: a fasta string
        output: a list of sequences from the fasta file
    """
    inp = io.StringIO(fasta_string)
    return parse_fasta(inp, return_names)


def write_sequential_fasta(path, sequences):
    """
        writes a fasta file to path, where the sequences are named as integers from 0 to len(sequences) - 1.
    """
    fasta_out, input_type = _open_if_is_name(path, "w")
    for i, seq in enumerate(sequences):
        print(f">{i}\n{seq}",file=fasta_out)
    if input_type == "name":
        fasta_out.close()

def write_partitioned_fasta(path, sequences):
    """
        sequences is a dict where keys are categories of sequences and values are lists of sequences:

        writes a fasta file to path, where the sequences are named as category_[0-9]+ .
    """
    with open(path,"w") as fasta_out:
        for category, seqs in sequences.items():
            for i, seq in enumerate(seqs):
                print(f">{category}_{i}\n{seq}",file=fasta_out)

def generate_alignment(sequences, ep=0.0, op=1.53):
    """
        uses mafft to align sequences.
        
        sequences is a dict where keys are categories of sequences and values are lists of sequence
        
        
        returns (seq_names, sequences) as a tuple of lists.

    """
    with tempfile.TemporaryDirectory() as tmp:
        #tmp_fasta_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
        tmp_fasta_path =  tmp + "/tmp.fasta" #str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
        # tmp_fasta_out_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
        write_partitioned_fasta(tmp_fasta_path,sequences)
        align_out = subprocess.run(['mafft', '--thread', '8', '--maxiterate', '1000', '--globalpair', "--ep", str(ep), "--op", str(op), tmp_fasta_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            align_out.check_returncode()
        except:
            print(align_out.stderr, sys.stderr)
            raise(Exception)
    return parse_fasta_string(align_out.stdout.decode('utf-8'),True)

def run_phmmer(query, database, evalue=10, cpu=2):
    """
    Takes a <query> list of protein sequences,
        run hmmscan against the <database>,
            returns the best hit or hits for each
            coding sequence if no hits, returns None
    Args:
        genbank: str, path-like
            The input genbank file (annotated genome sequence)

        query: 
            string of a protein sequence

        database: str, path-like
            protein fasta file

        evalue: float
            The threshold E value for the phmmer hit to be reported

        cpu: float
            The number of CPU cores to be used to run phmmer

    Returns: a list of hits ranked by how good the hits are.

  
    """
    # Create a fasta file containing the query protein sequence. The fasta file name is based on input genbank file name
    with tempfile.TemporaryDirectory() as tmp:
        queryfa_path = tmp + "/query.fa"
        query_name = "QUERY"
        with open(queryfa_path, "w") as tmpfasta:
            print(f">{query_name}\n{query}", file=tmpfasta)
        
        
        search_args = ['phmmer', '--noali', '--notextw', '--cpu', str(cpu), '-E', str(evalue)]
        search_args += [queryfa_path, database]
        out = subprocess.run(search_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             encoding='utf-8')

        if out.returncode != 0:
            print(
                f'Error in hmmer execution: \n{out.stdout}\n{out.stderr}', file=sys.stderr)
            exit(1)

        hits = SearchIO.read(io.StringIO(out.stdout), 'hmmer3-text')
        hit_names = [x.id for x in hits]
        
    return hit_names

class SequenceSubsetter:
    subset_strategies = {"random","in_order"}

    @classmethod
    def subset(cls, seq_list: list, n: int, keep_first: bool = False, strategy: str = "random", random_seed: int = None) -> list:
        """
            input:
                seq_list: a list of protein sequence strings
                n: how many members of seq_list to copy into the output. If n > len(seq_list), a copy of seq_list will be returned
                keep_first: if set, then copy seq_list[0] into output and sample n-1 additional items
                strategy: 
                    "random": take sequences randomly from seq_list (without replacement)
                    "in_order": take the top n sequences from seq_list
                random_seed: provided to the random number generator.
                
                Not implemented:
                delete_end_gaps: if supplied then truncates the alignment by deleting all positions before the first non-gap position in any sequnence and after the last non-gap position in any sequence.
        """
        #TODO: implement delete_end_gaps

        output = list()
        if n <= 0:
            return output

        tmp_list = seq_list.copy()
        if keep_first:
            output.append(seq_list[0])
            n = n - 1
            tmp_list = tmp_list[1:]

        if strategy not in cls.subset_strategies:
            raise ValueError(f"sampler strategy {strategy} not recognized, must be one of {cls.subset_strategies}")
        elif strategy == "random":
            random.Random(random_seed).shuffle(tmp_list)
            output += tmp_list[0:n]
        elif strategy == "in_order":
            output += tmp_list[0:n]
        
        # if delete_end_gaps:
        #     output = delete_msa_endgaps(output)
        return output
