import enum
import pytest
from pgen import models, esm_msa_sampler, utils
from pgen import pgen_msa_revised
from io import StringIO
from pgen.esm_msa_sampler import ESM_MSA_ALLOWED_AMINO_ACIDS
import tempfile


def test_pgen_msa_revised_1(shared_datadir):
    with tempfile.TemporaryDirectory() as output_dir:
        #output_dir = "tmp_out"
        outpath = output_dir + "/generated.fasta"
        pgen_msa_revised.main(["--templates",str(shared_datadir / "test_query_seqs.fasta"), "--references", str(shared_datadir / "test_reference_seqs.fasta"), "-o", outpath, "--alignment_size", "1", "--seqs_per_template", "2"])
        outseq_names, outseqs = utils.parse_fasta(outpath, return_names=True)
        assert outseq_names == ["0_query_seq1", "1_query_seq1", "0_query_seq2", "1_query_seq2"]
        assert len(outseqs) == 4
        assert len(outseqs[0]) == 5
        assert len(outseqs[1]) == 5
        assert len(outseqs[2]) == 6
        assert len(outseqs[3]) == 6

