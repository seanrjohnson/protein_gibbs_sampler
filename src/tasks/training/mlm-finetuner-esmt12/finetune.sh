python finetune.py \
    --model esm1_t12_85M_UR50S \
    --arch protein_bert_base \
    --checkpoint_interval 10 \
    --datapath /workspace/data/tautomerase_2953.fasta \
    --epochs 100 \
    --output_dir /workspace/outputs \
    --toks_per_batch 4096

