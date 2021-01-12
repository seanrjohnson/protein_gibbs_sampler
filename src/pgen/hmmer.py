#Functions for checking how well sequences match to hmm profiles.
#requires hmmer3 to be installed.
import uuid
from pathlib import Path
import subprocess
from collections import OrderedDict
from pgen import utils

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

class HmmParser:
    #From srj_chembiolib
    def __init__(self, filename, filetype="tblout"):
        if filetype=="domtblout":
            self.header = ["target name", "target accession", "target length", "query name", "query accession", "query length", "full sequence E-value", "full sequence score", "full sequence bias", "this domain #", "this domain of", "this domain c-Evalue", "this domain i-Evalue", "this domain score", "this domain bias", "hmm coord from", "hmm coord to", "ali coord from", "ali coord to", "env coord from", "env coord to", "acc", "description of target"]
        else:
            self.header = ["target name", "target accession", "query name", "query accession", "full sequence E-value", "full sequence score", "full sequence bias", "best domain E-value", "best domain score", "best domain bias", "exp", "reg", "clu", "ov", "env", "dom", "rep", "inc", "description of target"]

        (self.infile, self.input_type) = _open_if_is_name(filename)

    def __iter__(self):
        return self

    def __del__(self):
        if (self.input_type == "name"):
            self.infile.close()

    def __next__(self):
        line = self.infile.readline()
        while (line != ''):
            line = line.strip()
            if ((len(line) == 0) or (line[0] == "#")) :
                line = self.infile.readline()
            else:
                rec = OrderedDict()
                parts = line.split(None,len(self.header)-1)
                for i,k in enumerate(self.header):
                    rec[k] = parts[i]
                return rec

        if (self.input_type == "name"):
            self.infile.close()
        raise StopIteration

def get_hmm_scores(profile_path, sequences, tmp_dir="/tmp", default_score=0, score_type="full sequence score"):
    """
        Note that this function generates temporary files that may have overlaping names between
        calls, so it should not be considered thread safe.
    
        input:
            - profile_path: path to a .hmm file containing a single hmmer profile. Behavior of this function is undefined in cases where 
                            the .hmm file contains multiple profiles. #TODO: handle the multiple profile case.
            - sequences: a list of strings of protein sequences
            - tmp_dir: where temporary files like fasta files and tblout files will be written. (this function will not delete them, you must delete them manually if you want to clean them)
            - default_score: for sequences with no matches (where they fall below the filter threshold of hmmscan), they will be given this score. Some sensible values might be 0, or None.
            - score_type: which column from the tblout file to report
    """
    pp = Path(profile_path)
    out = None
    if not pp.exists():
        raise(ValueError(f"{profile_path} does not exist"))
    #print(pp.with_suffix(p.suffix+'.h3i'))
    if not pp.with_suffix(pp.suffix+'.h3i').exists(): # the hmm profile has not been pressed
        out = subprocess.run(['hmmpress',str(pp)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out.check_returncode()
        except:
            print(out.stderr)
            raise(Exception)
    tmp_fasta_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
    utils.write_sequential_fasta(tmp_fasta_path,sequences)
    tblout_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".tblout"))
    
    #set the thresholds really high so that we get as many hits as possible
    hmmscan_out = subprocess.run(['hmmscan',"-E","1000000", "--domE", "10000000" ,"--tblout", tblout_path, profile_path, tmp_fasta_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        hmmscan_out.check_returncode()
    except:
        print(hmmscan_out.stderr)
        raise(Exception)
    hmm_parser = HmmParser(tblout_path)
    
    out = [default_score] * len(sequences)
    for rec in hmm_parser:
        i = int(rec["query name"])
        f = float(rec[score_type])
        if out[i] == default_score:
            out[i] = f
        else:
            raise(Exception("Duplicates in tblout!"))
    return out
        
    
