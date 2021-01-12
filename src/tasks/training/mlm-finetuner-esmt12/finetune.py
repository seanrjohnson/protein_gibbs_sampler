from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
import os
import pathlib
import torch
from esm.pretrained import load_model_and_alphabet

# Custom
from pgen.core import ModelNames
from pgen.utils import setup_logger
from pgen.data import MaskingBatchConverter, load_train_splits_from_fasta
from pgen.tokenizers import CharacterTokenizer
from pgen.trainers import ESMTrainer

logger = logging.getLogger(__name__)

def create_parser():
    parser = ArgumentParser(description="Run training for given model")

    parser.add_argument(
        "--datapath",
        type=pathlib.Path,
        help="FASTA file on which to run training",
        default=f"/workspace/data/fasta/project/Russ_chorismate_reference_seqs.fasta"
    )
    
    parser.add_argument(
        "--arch",
        type=str,
        help="Model architecture -- 'protein_bert_base' 'roberta_large'",
        default="protein_bert_base")
    
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
        default="local/data/finetune_output"
    )

    # ESM used 4096 for extract
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )

    parser.add_argument("--adam_betas", type=float, nargs='+', default=[0.9, 0.999])
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    # parser.add_argument("--best_checkpoint_metric", type=str, default="loss")
    # parser.add_argument("--bpe",default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint creation interval")
    # parser.add_argument("--criterion", type=str, default='protein_bert_loss')
    # parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=-1)
    # parser.add_argument("--gpt2_init", type=bool, default=True)
    # parser.add_argument("--learned_pos", type=bool, default=False)
    # parser.add_argument("--lr_scheduler", type=str, default='inverse_sqrt')

    # The learning rate we used in the paper was 1e-4. However, if you are doing additional steps of pre-training starting from an existing BERT checkpoint, you should use a smaller learning rate (e.g., 2e-5).
    parser.add_argument("--lr", nargs='+', type=float, default=[2e-5]) # was 1e-3
    # parser.add_argument("--mask_ratio", type=float, default=0.15)
    # parser.add_argument("--mask_rnd", type=float, default=0.1)
    # parser.add_argument("--mask_same", type=float, default=0.1)
    # parser.add_argument("--max_epoch", type=int, default=100)
    # parser.add_argument("--max_tokens", type=int, default=1025) 
    # parser.add_argument("--max_positions", type=int, default=1024)
    # parser.add_argument("--min_lr", type=float, default=1e-09)
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    # parser.add_argument("--optimizer", type=str, default='adam')
    # parser.add_argument("--place_mask_strategy", type=str, default='random')
    # parser.add_argument("--task", type=str, default='protein_bert')
    # parser.add_argument("--tokens_per_sample", type=int, default=512, help="Maximum batch size") # 1024
    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    # parser.add_argument("--token_dropout", type=int, default=100)
    # parser.add_argument("--update_freq", nargs='+', type=int, default=[8])
    # parser.add_argument("--use_eos", type=bool, default=False)
    # parser.add_argument("--validate_interval", type=int, default=1)
    # parser.add_argument("--warmup_init_lr", type=float, default=1e-07)
    # parser.add_argument("--warmup_updates", type=int, default=16000)
    parser.add_argument("--model", type=ModelNames, default=ModelNames.DEFAULT)
    
    return parser

def main(args):
    # -------------------------------------------------------------------
    # Load Model
    # -------------------------------------------------------------------
    model_name = args.model # ModelNames.T6_43M_UR50S # ModelNames.DEFAULT
    logger.info(f'Attempting to load ESM Model {model_name}...')
    model, alphabet = load_model_and_alphabet(model_name.value)
    logger.debug(model)
    logger.info(f'Loaded ESM Model {model_name}')
    
    # -------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------
    vars(model.args).update(vars(args))
    vars(args).update(vars(model.args))
    logger.info(f"Training Arguments: {args}")
    device = "cpu" if args.nogpu or not torch.cuda.is_available() else "cuda"

    # ------------
    # system setup
    # ------------
    output_path = pathlib.Path(os.path.join(args.output_dir, model_name.value, datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'Output logging to: {output_path}')
    
    # -------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------
    tokenizer = CharacterTokenizer()
    batch_converter = MaskingBatchConverter(alphabet, tokenizer)
    datacontainer = load_train_splits_from_fasta(args.datapath, batch_converter, args)

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    trainer = ESMTrainer(model_name, model, datacontainer, args, output_path, alphabet, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr[0], betas=args.adam_betas, eps=args.adam_eps)#weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # Ignore index important because we set unmasked token labels to -1, so ignored in loss

    res = trainer.train(
        optimizer=optimizer,
        criterion=criterion
    )

    logger.info(f'Finished {args.epochs}/{args.epochs} epochs.')

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)

    root_logger = logging.getLogger()
    setup_logger(root_logger, args.output_dir)

    