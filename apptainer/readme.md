# To build
`apptainer build apptainer.sif apptainer.def`


# To test:
```bash
apptainer shell --nv apptainer.sif
cd /protein_gibbs_sampler
. ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base
pytest test
```

# To run scripts
```bash
apptainer exec --nv apptainer.sif pgen_esm.py
```
note: `apptainer run` seems to work also
