# To build
`apptainer build singularity.sif esm.def`


# To test:
```bash
apptainer shell --nv singularity.sif
cd /protein_gibbs_sampler
. ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base
pytest test
```

# To run scripts
```bash
apptainer exec --nv singularity.sif pgen_esm.py
```
Note: apptainer run seems to work also.
