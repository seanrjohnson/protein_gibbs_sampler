python run_mlm.py \
# Don't use if training from scratch
    # --model_name_or_path roberta-base \
# If training from scratch, pass a model type from the list: layoutlm, distilbert, albert, bart, camembert, xlm-roberta, longformer, roberta, squeezebert,
#                     bert, mobilebert, flaubert, xlm, electra, reformer, funnel
    --model_type roberta \
# Pretrained config name or path if not the same as model_name
    # --config_name
# Pretrained tokenizer name or path if not the same as model_name
    --tokenizer_name Rostlab/prot_bert_bfd \ # TODO -- ok to use protbert one?
    --cache_dir /workspace/cache/hf_transformers \
    --train_file /workspace/data/train.fasta \
    --validation_file /workspace/data/validation.fasta \
    # --overwrite_cache # Overwrites cached training and eval sets
# The max total input sequence length after tokenization.
# Sequences longer than will be truncated
    --max_seq_length 110 \
    --preprocessing_num_workers 8 \
# Ratio of tokens to mask for masked language modeling loss
    --mlm_probability 0.15 \
# Whether distinct lines of text in dataset are to be handled as distinct sequences 
    --line_by_line
# Whether to pad all examples to 'max_seq_len'. If false, will pad samples dynamically when batching
# up to the max length in the batch
    # --pad_to_max_length
# Model predictions and checkpoints written to 
    --output_dir ./output
# Overwrite the content of the output directory.Use this to continue training if output_dir points to a checkpoint directory.
    # --overwrite_output_dir
    --do_train \
    --do_eval \
    --do_predict \
    # --evaluation_strategy \ # {EvaluationStrategy.NO,EvaluationStrategy.STEPS,EvaluationStrategy.EPOCH}
    --prediction_loss_only
    --per_device_train_batch_size 32
    --per_device_eval_batch_size 64
    # --gradient_accumulation_steps ??
    # --eval_accumulation_steps ??
    --learning_rate 1e-3 \ # TODO -- Tune
    --weight_decay ?? \
    --adam_beta1 ?? \
    --adam_beta2 ?? \
    --adam_epsilon ?? \
    --num_train_epochs 100 \
    # --max_steps ?? \
    --warmup_steps 1000 \ # Best guess ??
    --logging_dir ./logs \
    --logging_first_step True \ # NOT SURE IF THIS IS BOOL? Log the first global_step
    --logging_steps 100 \ # Log every x steps
    --save_steps 100 \ # Checkpoints every x steps TODO What determines a step?
    --save_total_limit 3 \ 
    # --no_cuda \ 
    --seed 42 \
    --fp16 \
    --fp16_opt_level '02' \
    # --dataloader_drop_last
    --eval_steps 100 \ # TODO Guessing
    --dataloader_num_workers 8 \
    # --past_index
    --run_name roberta_initial_run \
    # --label_names
    --load_best_model_at_end True \
    --metric_for_best_model 'eval_loss' \
    --greater_is_better False