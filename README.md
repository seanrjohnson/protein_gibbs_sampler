# Generating novel protein sequences using Gibbs sampling of masked language models

This repository represents the code supporting the work done in [Generating novel protein sequences using Gibbs sampling of masked language models
](https://www.biorxiv.org/content/10.1101/2021.01.26.428322v1.full).
Since then, the code has been continuously updated. For the version of the code that was used in that preprint, see: [here](https://github.com/seanrjohnson/protein_gibbs_sampler/tree/v0.1.0)

## Install

Clone the repo and move into the new directory

```bash
git clone https://github.com/seanrjohnson/protein_gibbs_sampler.git
cd protein_gibbs_sampler
```

Then install either through conda

### Conda/Pip

Make a new conda environment called `protein_gibbs_sampler`
```bash
conda env create --name protein_gibbs_sampler -f conda_env.yml
```

Test the install
```bash
conda activate protein_gibbs_sampler
pytest .
```

If you have a CUDA compatible GPU, everything should pass, otherwise there will be one skipped test.

Running the tests for the first time might take a while because model weights need to be downloaded.

## Generating new protein sequences from the command line

This package contains four command line programs to make it easy to generate new sequences.
For good performance, it is recommended to use GPU, but they will still run on a CPU, just excruciatingly slow for everything but small proteins.

### pgen_esm.py

Given a seed sequence, generates new sequences.

#### pgen_esm_input.tsv
```tsv
test_seq	{'num_iters': 20, 'burnin': 10, 'mask': True, 'in_order':False, 'num_positions_percent': 10, 'seed_seq': "MEPAATGQEAEECAHSGRGEAWEEV"}
test_seq2	{'num_iters': 20, 'burnin': 10, 'mask': True, 'in_order':False, 'num_positions_percent': 10, 'seed_seq': "MLEGADIVIIPAGV"}
```

```bash
pgen_esm.py -o pgen_out -i pgen_esm_input.tsv --num_output_sequences 10
```

For detailed help:
`pgen_esm.py -h`

### pgen_esm_from_fasta.py

Like `pgen_esm.py` except that the seed sequences come from fasta files, instead of being defined in the sampler arguments. The sequences in the fasta can either be aligned or unaligned. If they are aligned, then gaps will be removed before sampling, but setting `--keep_gap_positions` will add the gaps back in after sampling. If the fasta contains more than one sequence, then a random sequence will be selected for each round of sampling.

```bash
pgen_esm_from_fasta.py -o pgen_from_fasta_out -i pgen_msa_input.tsv --num_output_sequences 10 --keep_gap_positions
```

For detailed help:
`pgen_esm_from_fasta.py -h`


### pgen_msa.py

Given a seed msa, uses esm-msa to generate new sequences.

This program will randomly mask and resample across the entire MSA.

#### fasta_input1.fasta
```fasta
>s1
MEPAATGQEAE--AHSGRGEAWEEV
>s2
MCP-ATGR-AEMCAHS--GEAWLLV
>s3
MEQ-AGGRLAEM-AHHC-GEAWLLV
```

#### fasta_input2.fasta
```fasta
>s1
MLEGADIVIIP-GV
>s2
MLDG---VLLPGAV
>s3
M-EPADILVV--GV
```

#### pgen_msa_input.tsv
```tsv
test_seq	{'num_iters': 20, 'burnin': 10, 'mask': True, 'in_order':False, 'num_positions_percent': 10}	fasta_input1.fasta
test_seq2	{'num_iters': 20, 'burnin': 10, 'mask': True, 'in_order':False, 'num_positions_percent': 10}	fasta_input2.fasta
```

```bash
pgen_msa.py -o pgen_msa_out -i pgen_esm_msa_input.tsv --num_output_sequences 10 
```

For detailed help:
`pgen_msa.py -h`

### pgen_msa_revised.py

This program will randomly mask and resample a single sequence from the MSA. It can also generate new MSAs from a the top phmmer hits in a list of reference sequences.

example

```bash
pgen_msa_revised.py --templates list_of_sequences_to_resample.fasta --references large_list_of_reference_sequences.fasta -o generated_sequences.fasta --device gpu --alignment_size 32 --passes 3 --burn_in 2
```

For detailed help and additional options:
`pgen_msa_revised.py -h`

## Calculating sequence probabilities

ESM and ESM-MSA models can be used to assign probabilities to protein sequences and individual positions within protein sequences

### likelihood_esm.py

Use the single-sequence ESM models to calculate sequence likelihoods.

```bash
likelihood_esm.py -i {input} -o {output} --model esm1v --csv --device gpu --score_name esm1v-mask6 --mask_distance 6 --positionwise positionwise_sequence_probabilities.csv
```

For detailed help and additional options:
`likelihood_esm.py -h`

### likelihood_esm_msa.py

Use the mutliple sequence alignment ESM model to calculate sequence likelihoods.

We have noticed that the probability scores from ESM-MSA are best from partially masked sequences. Masking the entire target sequence works very poorly. Not masking any positions gives higher than expected scores for highly divergent targets.


Note that in some cases `--reference_msa` can actually be an unaligned fasta file, because the sequences may be re-aligned at runtime, depending on the `subset_strategy`.

```bash
likelihood_esm_msa.py -i seqs_to_calculate_probabilities_for.fasta -o whole_sequence_probabilities.csv --reference_msa reference_sequences.fasta --device gpu --subset_strategy top_hits --alignment_size 31 --count_gaps --mask_distance 6 --csv --positionwise positionwise_sequence_probabilities.csv
```
For detailed help and additional options:
`likelihood_esm_msa.py -h`


## References

This repository represents work building on related resources as cited below.

### bert-gen

[Github](https://github.com/nyu-dl/bert-gen)

[Paper](https://arxiv.org/abs/1902.04094)

```bibtex
@article{wang2019bert,
  title={BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model},
  author={Wang, Alex and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1902.04094},
  year={2019}
}
```

### ESM

[Github](https://github.com/facebookresearch/esm)

[Paper](https://doi.org/10.1101/622803)

```bibtex
@article{rives2019biological,
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
  title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
  year={2019},
  doi={10.1101/622803},
  url={https://www.biorxiv.org/content/10.1101/622803v3},
  journal={bioRxiv}
}
```

### ESM-MSA

[Paper](https://doi.org/10.1101/2021.02.12.430858)

```bibtex
@article{rao2021msa,
  author = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
  title={MSA Transformer},
  year={2021},
  doi={10.1101/2021.02.12.430858},
  url={https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1},
  journal={bioRxiv}
}
```

