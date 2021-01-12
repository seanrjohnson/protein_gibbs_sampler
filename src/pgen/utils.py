from .core import SingletonMeta
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict
import io
import pandas as pd
from pathlib import Path
import uuid
import subprocess

logger = logging.getLogger(__name__)

class CommonVars(metaclass=SingletonMeta):
    def __init__(self, overrides: Dict[str, any]={}) -> None:
        self.app = os.getenv('APPNAME', 'PGEN')
        self.workspace = os.getenv(f'{self.app}_WORKSPACE', '/workspace')
        self.project_dir = f'{self.workspace}'
        self.data_dir = f'{self.workspace}/data'
        self.logs_dir = f'{self.workspace}/logs'

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        self.__dict__.update(overrides)
        logger.debug(self.__repr__())
        
    def __repr__(self):
        return f"{self.__dict__}"

def get_common() -> CommonVars:
    return CommonVars()

def setup_logger(app_logger, output_dir="/workspace/logs", log_level=logging.INFO):
    standard_log_level = log_level
    app_logger.setLevel(standard_log_level)

    log_formatter = logging.Formatter('[%(levelname)s][%(name)s] %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logging_stream_handler = logging.StreamHandler()
    logging_stream_handler.setLevel(standard_log_level)
    logging_stream_handler.setFormatter(log_formatter)
    app_logger.addHandler(logging_stream_handler)

    overrides = {} # {'workspace': 'test'}

    logging_file_handler = RotatingFileHandler(
        filename=f"{output_dir}/pgen.log", 
        maxBytes=2e6,
        backupCount=1)
    logging_file_handler.setLevel(standard_log_level)
    logging_file_handler.setFormatter(log_formatter)
    app_logger.addHandler(logging_file_handler)

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


def parse_fasta(filename, return_names=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
        input: the name of a fasta file or a filehandle to a fasta file
        output: a list of sequences from the fasta file
    """
    
    prev_len = 0
    prev_name = None
    prev_seq = ""
    out_seqs = list()
    out_names = list()
    (input_handle, input_type) = _open_if_is_name(filename)

    for line in input_handle:
        line = line.strip()
        
        if line[0] == ">":
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
    if return_names:
        return out_names, out_seqs
    else:
        return out_seqs

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

def first_order_statistics(prefixes, names, seqs):
    """
        input: 
            prefixes: a list of strings, such that each sequence name starts with one of the prefixes
            names: list of sequence names
            seqs: list of aligned sequences. Must all be of same length.
        output: 
            a data frame with index=['position','AA'], columns='prefix'
            values are frequencies
    """
    
    out_dict = dict() #{position: {AA: {prefix: [count] }}}
    prefix_counts = {x:0 for x in prefixes}
    
    for i,name in enumerate(names):
        prefix = None
        for prfx in prefixes:
            if name.startswith(prfx):
                prefix = prfx
                prefix_counts[prefix] += 1
        if prefix is None:
            raise(Exception(f"{name} does not have valid prefix"))
        for pos_1,aa in enumerate(seqs[i]):
            if pos_1 not in out_dict:
                out_dict[pos_1] = dict()
            if aa not in out_dict[pos_1]:
                out_dict[pos_1][aa] = {prfx: 0 for prfx in prefixes}
            out_dict[pos_1][aa][prefix] += 1
    out_list = list()
    
    for position, AAs in out_dict.items():
        for AA, prefxes in AAs.items():
            for prefix, count in prefxes.items():
                out_list.append({"position": position, "AA": AA, "prefix": prefix, "frequency": count/prefix_counts[prefix]})

    df = pd.DataFrame(out_list)

    return df.pivot(index=['position','AA'], columns='prefix')['frequency']

def second_order_statistics(prefixes, names, seqs):
    """
        input: 
            prefixes: a list of strings, such that each sequence name starts with one of the prefixes
            names: list of sequence names
            seqs: list of aligned sequences. Must all be of same length.
        output: 
            a data frame with index=['position_1', 'position_2', 'AA1', 'AA2'], columns='prefix'
            values are frequencies
    """
    
    out_dict = dict() #{position: {AA: {prefix: [count] }}}
    prefix_counts = {x:0 for x in prefixes}
    
    for i,name in enumerate(names):
        prefix = None
        for prfx in prefixes:
            if name.startswith(prfx):
                prefix = prfx
                prefix_counts[prefix] += 1
        if prefix is None:
            raise(Exception(f"{name} does not have valid prefix"))
        for pos_1, aa1 in enumerate(seqs[i]):
            
            if pos_1 not in out_dict:
                out_dict[pos_1] = dict()
            if aa1 not in out_dict[pos_1]:
                out_dict[pos_1][aa1] = dict()
            
            for pos_2 in range(pos_1+1, len(seqs[i])):
                aa2 = seqs[i][pos_2]
                if pos_2 not in out_dict[pos_1][aa1]:
                    out_dict[pos_1][aa1][pos_2] = dict()
                if aa2 not in out_dict[pos_1][aa1][pos_2]:
                    out_dict[pos_1][aa1][pos_2][aa2] = {prfx: 0 for prfx in prefixes}
                out_dict[pos_1][aa1][pos_2][aa2][prefix] += 1
                
            
    out_list = list()
    
    for position1, AA1s in out_dict.items():
        for AA1, position2s in AA1s.items():
            for position2, AA2s in position2s.items():
                for AA2, prefxes in AA2s.items():
                    for prefix, count in prefxes.items():
                        out_list.append({"position_1": position1, "position_2":position2, "AA1": AA1, "AA2": AA2, "prefix": prefix, "frequency": count/prefix_counts[prefix]})

    df = pd.DataFrame(out_list)

    return df.pivot(index=['position_1', 'position_2', 'AA1','AA2'], columns='prefix')['frequency']

def third_order_statistics(prefixes, names, seqs):
    """
        input: 
            prefixes: a list of strings, such that each sequence name starts with one of the prefixes
            names: list of sequence names
            seqs: list of aligned sequences. Must all be of same length.
        output: 
            a data frame with index=['position_1', 'position_2', 'position_3', 'AA1', 'AA2', 'AA3'], columns='prefix'
            values are frequencies
    """
    
    out_dict = dict() #{position: {AA: {prefix: [count] }}}
    prefix_counts = {x:0 for x in prefixes}
    
    for i,name in enumerate(names):
        prefix = None
        for prfx in prefixes:
            if name.startswith(prfx):
                prefix = prfx
                prefix_counts[prefix] += 1
        if prefix is None:
            raise(Exception(f"{name} does not have valid prefix"))
        for pos_1, aa1 in enumerate(seqs[i]):
            
            if pos_1 not in out_dict:
                out_dict[pos_1] = dict()
            if aa1 not in out_dict[pos_1]:
                out_dict[pos_1][aa1] = dict()
            
            for pos_2 in range(pos_1+1, len(seqs[i])):
                aa2 = seqs[i][pos_2]
                if pos_2 not in out_dict[pos_1][aa1]:
                    out_dict[pos_1][aa1][pos_2] = dict()
                if aa2 not in out_dict[pos_1][aa1][pos_2]:
                    out_dict[pos_1][aa1][pos_2][aa2] = dict()
                
                for pos_3 in range(pos_2+1,len(seqs[i])):
                    aa3 = seqs[i][pos_3]
                    if pos_3 not in out_dict[pos_1][aa1][pos_2][aa2]:
                        out_dict[pos_1][aa1][pos_2][aa2][pos_3] = dict()
                    if aa3 not in out_dict[pos_1][aa1][pos_2][aa2][pos_3]:
                        out_dict[pos_1][aa1][pos_2][aa2][pos_3][aa3] = {prfx: 0 for prfx in prefixes}
                    out_dict[pos_1][aa1][pos_2][aa2][pos_3][aa3][prefix] += 1
                
            
    out_list = list()
    #print(out_dict)
    for position1, AA1s in out_dict.items():
        for AA1, position2s in AA1s.items():
            for position2, AA2s in position2s.items():
                for AA2, position3s in AA2s.items():
                    for position3, AA3s in position3s.items():
                        for AA3, prefxes in AA3s.items():
                            for prefix, count in prefxes.items():
                                out_list.append({"position_1": position1, "position_2": position2, "position_3": position3, "AA1": AA1, "AA2": AA2, "AA3": AA3, "prefix": prefix, "frequency": count/prefix_counts[prefix]})

    df = pd.DataFrame(out_list)

    return df.pivot(index=['position_1', 'position_2', 'position_3', 'AA1', 'AA2', 'AA3'], columns='prefix')['frequency']

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
    align_out = subprocess.run(['mafft', tmp_fasta_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        align_out.check_returncode()
    except:
        print(align_out.stderr)
        raise(Exception)
    return parse_fasta_string(align_out.stdout.decode('utf-8'),True)
