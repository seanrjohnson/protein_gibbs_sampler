from .core import SingletonMeta
import os
import logging
from logging.handlers import RotatingFileHandler
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

class RawAndDefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def _open_if_is_name(filename_or_handle):
    out = filename_or_handle
    input_type = "handle"
    try:
        out = open(filename_or_handle,"r")
        input_type = "name"
    except TypeError:
        pass
    except Exception as e:
        raise(e)

    return (out, input_type)

LEGAL_AA_CODES={c:i for i,c in enumerate("-ACDEFGHIKLMNOPQRSTUVWYX")} #mapping characters to indexes
AA_REVERSE_LOOKUP=len(LEGAL_AA_CODES)*[""]
for i in LEGAL_AA_CODES:
    AA_REVERSE_LOOKUP[LEGAL_AA_CODES[i]] = i #mapping indexes to characters

def msa_to_matrix(list_of_strings):
    """converts an msa to a numerical array of dimensions num_sequences x alignment length"""
    msa = list_of_strings
    tr = LEGAL_AA_CODES
    out = np.zeros((len(msa), len(msa[0])), dtype=np.uint8)
    for i in range(len(msa)):
        for j in range(len(msa[0])):
            out[i,j] = tr[msa[i][j]]
    return out


def msa_to_frequencies(inp_names, inp_seqs, description_prefix=None):
    '''
        Converts a multiple sequence alignment (msa) (a list of sequences) into an array of dimensions number_of_legal_aa_characters x sequence_length
        where the values are in the range (0,1) and sum to 1 for each column

        description_prefix can be used to select a subset, based on name, of the sequences in the input msa.

        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/correlated_mutations.py
    '''
    seqs = list()

    for i in range(len(inp_seqs)):
        if ( (description_prefix is None) or (inp_names[i].startswith(description_prefix)) ):
            seqs.append(inp_seqs[i])

    out = np.zeros((len(LEGAL_AA_CODES), len(inp_seqs[0]) ))
    
    msa_matrix = msa_to_matrix(seqs) # num_seqs x seq_len = AA_idx

    for i in range(msa_matrix.shape[0]):
        for pos in range(msa_matrix.shape[1]):
            out[msa_matrix[i,pos],pos] += 1
    return out.astype(np.float64) / np.float64(len(seqs))

def msa_to_second_order_statistics(inp_names, inp_seqs, description_prefix=None):
    '''
      calculates raw correlations between positions in an msa_array

      output: an array of dimensions (alignment_length, alignment_length, AA_CODES_length, AA_codes_length)
      where the indexes are: (first_position_AA, second_position_AA,first_position_index, second_position_index)
      and the values are the frequency of the associations (0-1)
        description_filter can be used to select a subset, based on name, of the sequences in the input msa.

    adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/correlated_mutations.py
    '''
    seqs = list()

    for i in range(len(inp_seqs)):
        if ( (description_prefix is None) or (inp_names[i].startswith(description_prefix)) ):
            seqs.append(inp_seqs[i])
    
    msa_matrix = msa_to_matrix(seqs) # num_seqs x seq_len = AA_idx

    out = np.zeros((len(LEGAL_AA_CODES),len(LEGAL_AA_CODES),msa_matrix.shape[1],msa_matrix.shape[1]), dtype=np.uint32)
    for i in range(msa_matrix.shape[0]):
      # print(seq_number)
      for pos1 in range(msa_matrix.shape[1]):
        for pos2 in range(msa_matrix.shape[1]):
          out[msa_matrix[i,pos1], msa_matrix[i,pos2], pos1, pos2] += 1 

    return out.astype(np.float64)/np.float64(len(seqs))

def second_order_correlations(fos, sos):
    """
        sos[aa1,aa2,pos1,pos2] - fos[aa1,pos1]*fos[aa2,pos2]
    """

    # TODO: try to vectorize this.
    out = np.zeros(sos.shape)

    for aa1 in range(sos.shape[0]):
        for aa2 in range(sos.shape[1]):
            for pos1 in range(sos.shape[2]):
                for pos2 in range(sos.shape[3]):
                    out[aa1,aa2,pos1,pos2] = sos[aa1,aa2,pos1,pos2] - (fos[aa1,pos1] * fos[aa2,pos2])
    return out

def flatten_second_order(sos):
    """
        sos will be symmetric, so we only need one hez
    """
    out = np.zeros(sos.shape[2]*sos.shape[2])
    return out

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

        output: sequences or names, sequences
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
    with open(path,"w") as fasta_out:
        for i, seq in enumerate(sequences):
            print(f">{i}\n{seq}",file=fasta_out)

def write_partitioned_fasta(path, sequences):
    """
        sequences is a dict where keys are categories of sequences and values are lists of sequences:

        writes a fasta file to path, where the sequences are named as category_[0-9]+ .
    """
    with open(path,"w") as fasta_out:
        for category, seqs in sequences.items():
            for i, seq in enumerate(seqs):
                print(f">{category}_{i}\n{seq}",file=fasta_out)

def generate_alignment(sequences, tmp_dir="/tmp"):
    """
        uses mafft to align sequences.
        
        sequences is a dict where keys are categories of sequences and values are lists of sequence
        
        
        returns (seq_names, sequences) as a tuple of lists.

    """

    #tmp_fasta_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
    tmp_fasta_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
    tmp_fasta_out_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
    write_partitioned_fasta(tmp_fasta_path,sequences)
    align_out = subprocess.run(['mafft', '--thread', '8', '--maxiterate', '1000', '--globalpair', tmp_fasta_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        align_out.check_returncode()
    except:
        print(align_out.stderr)
        raise(Exception)
    return parse_fasta_string(align_out.stdout.decode('utf-8'),True)

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
