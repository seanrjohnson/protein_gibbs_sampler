# Generating novel protein sequences using Gibbs sampling of masked language models

This repository represents the code supporting the work done in [Generating novel protein sequences using Gibbs sampling of masked language models
](https://www.biorxiv.org/content/10.1101/2021.01.26.428322v1.full)
Since then, the code has been continuously updated. For the version of the code that was used in that preprint, see: [here](https://github.com/seanrjohnson/protein_gibbs_sampler/tree/v0.1.0)

## Install

Clone the repo and move into the new directory

```bash
git clone https://github.com/seanrjohnson/protein_gibbs_sampler.git
cd protein_gibbs_sampler
```

Then install either through conda or Docker

### Conda/Pip

Make a clean new Conda environment
```bash
conda create -n protein_gibbs_sampler python~=3.8
conda activate protein_gibbs_sampler
```

Install this package and its prereqs with pip
```bash
pip install -e .
```

Test the install
```bash
pytest .
```

If you have CUDA installed, everything should pass, otherwise there will be one skipped test.

### Docker

Setup container environment:

```bash
# From root of this repo
# Create container -- starts container running in detached mode and root of 
# 	repo mounted to /workspace in the container
make run-cpu # cpu-only
# make run # GPU pass through

# Attach to container
make attach

# ---Runnining inside container---
# Pip install the source for this package
pip install -e .
```

Additional Commands:

```bash
# Additional Commands
make start
make stop
make shell
make remove
```

## Generating new protein sequences from the command line

This package contains three command line programs to make it easy to generate new sequences.
For good performance, it is recommended to use GPU, but they will still run on a CPU, just really slow for everything but small proteins.

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

### pgen_msa.py

Given a seed msa, uses esm-msa to generate new sequences.

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
pgen_msa.py -o pgen_msa_out -i pgen_esm_msa_input.tsv --num_output_sequences 10 --batch_size
```

For detailed help:
`pgen_msa.py -h`

### pgen_esm_from_fasta.py

Like `pgen_esm.py` except that the seed sequences come from fasta files, instead of being defined in the sampler arguments. The sequences in the fasta can either be aligned or unaligned. If they are aligned, then gaps will be removed before sampling, but setting `--keep_gap_positions` will add the gaps back in after sampling. If the fasta contains more than one sequence, then a random sequence will be selected for each round of sampling.

```bash
pgen_esm_from_fasta.py -o pgen_from_fasta_out -i pgen_msa_input.tsv --num_output_sequences 10 --keep_gap_positions
```

For detailed help:
`pgen_esm_from_fasta.py -h`


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

### ProTrans

[Github](https://github.com/agemagician/ProtTrans)
[Paper](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v2)

```
@article {Elnaggar2020.07.12.199554,
	author = {Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and BHOWMIK, DEBSINDHU and Rost, Burkhard},
	title = {ProtTrans: Towards Cracking the Language of Life{\textquoteright}s Code Through Self-Supervised Deep Learning and High Performance Computing},
	elocation-id = {2020.07.12.199554},
	year = {2020},
	doi = {10.1101/2020.07.12.199554},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Computational biology and bioinformatics provide vast data gold-mines from protein sequences, ideal for Language Models (LMs) taken from Natural Language Processing (NLP). These LMs reach for new prediction frontiers at low inference costs. Here, we trained two auto-regressive language models (Transformer-XL, XLNet) and two auto-encoder models (Bert, Albert) on data from UniRef and BFD containing up to 393 billion amino acids (words) from 2.1 billion protein sequences (22- and 112 times the entire English Wikipedia). The LMs were trained on the Summit supercomputer at Oak Ridge National Laboratory (ORNL), using 936 nodes (total 5616 GPUs) and one TPU Pod (V3-512 or V3-1024). We validated the advantage of up-scaling LMs to larger models supported by bigger data by predicting secondary structure (3-states: Q3=76-84, 8 states: Q8=65-73), sub-cellular localization for 10 cellular compartments (Q10=74) and whether a protein is membrane-bound or water-soluble (Q2=89). Dimensionality reduction revealed that the LM-embeddings from unlabeled data (only protein sequences) captured important biophysical properties governing protein shape. This implied learning some of the grammar of the language of life realized in protein sequences. The successful up-scaling of protein LMs through HPC to larger data sets slightly reduced the gap between models trained on evolutionary information and LMs. Availability ProtTrans: \&lt;a href="https://github.com/agemagician/ProtTrans"\&gt;https://github.com/agemagician/ProtTrans\&lt;/a\&gt;Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554},
	eprint = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554.full.pdf},
	journal = {bioRxiv}
}
```
