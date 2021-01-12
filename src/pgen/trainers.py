import datetime
import logging
import os
import pathlib
from uuid import uuid4
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

logger = logging.getLogger(__name__)
    
def tokens_to_seq(alphabet, batch):
        sequences = ["".join([alphabet.get_tok(idx) for idx in sequence]) for sequence in batch.cpu().numpy()]
        return sequences

class ESMTrainer:
    def __init__(self, model_name, model, data, args, output_dir, alphabet, device="cpu"):
        self.args = args

        _, self.train_dataloader, self.val_dataloader = data
        self.alphabet = alphabet
        self.output_dir = output_dir
        self.device = device
        self.model_name = model_name
        self.model = model
        self.model_args = self.parse_model_args(args)
        self.stopped_early = False

        self.num_train_examples = len(self.train_dataloader.dataset)
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_examples = len(self.val_dataloader.dataset)
        self.num_val_batches = len(self.val_dataloader)

        self.checkpoints_dir = pathlib.Path(os.path.join(output_dir, "checkpoints"))
        self.checkpoints_dir.mkdir(parents=True)

        run_date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        self.run_id = f"{run_date_str}_{str(uuid4()).split('-')[0]}"
        self.writer = SummaryWriter(log_dir=os.path.join("./runs", self.run_id, "train"))
        self.eval_writer = SummaryWriter(log_dir=os.path.join("./runs", self.run_id, "eval"))
        # with torch.no_grad():
        #     self.writer.add_graph(self.model, torch.randint(0, 32, (2, 90)), verbose=False)
        # self.writer.add_hparams(hparam_dict=vars(self.args), metric_dict={}, run_name='TestRunName')

    def parse_model_args(self, args):
        assert all(
            -(self.model.num_layers + 1) <= i <= self.model.num_layers for i in args.repr_layers
        )
        repr_layers = [
            (i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in args.repr_layers
        ]

        return dict(repr_layers=repr_layers)

    def checkpoint_training(self, filename, **kwargs):
        save_data = dict(
            model=self.model.state_dict(),
            args=self.model.args
        )

        if 'optimizer' in kwargs:
            save_data['optimizer'] = kwargs.get('optimizer').state_dict()
        
        torch.save(save_data, self.checkpoints_dir/filename)

    def train(self, optimizer, criterion):
        # Pull out training specific args
        epochs = self.args.epochs
        checkpoint_interval = self.args.checkpoint_interval
        repr_layers = self.model_args.get('repr_layers')

        # Send model to device
        self.model = self.model.to(self.device)
        
        global_step, last_step, last_epoch = 0, 0, 0
        best_eval_loss = float('inf')
        non_increasing_epochs = 0
        for epoch in trange(epochs, desc="Epoch"):
            loss_epoch = 0
            nb_tr_examples = 0
            train_step = 0
            for idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Iteration"):
                raw_labels, raw_seqs, tokens, labels = batch
                batch_size = len(raw_labels)            
                tokens = tokens.to(device=self.device, non_blocking=True)

                optimizer.zero_grad()
                out = self.model(tokens, repr_layers=repr_layers, return_contacts=False)
                logits = out["logits"].cpu()
                loss = criterion(logits.reshape(-1, self.model.alphabet_size), labels.view(-1))
                loss.backward()
                optimizer.step()

                # End of Iteration (batch)
                loss_epoch += loss.item()
                nb_tr_examples += batch_size
                train_step += 1
                global_step += 1
                self.writer.add_scalar('Loss', loss.item(), global_step)

            # End of Epoch
            loss_epoch /= train_step
            eval_result = self.eval(global_step)
            eval_loss = eval_result.get('eval_loss')
            
            # if ((epoch+1) % self.args.validate_interval == 0) ``

            if eval_loss < best_eval_loss:
                non_increasing_epochs = 0
                best_eval_loss = eval_loss
                filename = f"best-model-checkpoint-{self.run_id}.pt"
                self.checkpoint_training(filename=filename, optimizer=optimizer, epoch=epoch+1, global_step=global_step, stopped_early=False)
                self.writer.add_text("ModelCheckpoint", f"file: {filename}\neval_loss: {eval_loss}\nepoch:{epoch+1}", global_step)
            else:
                non_increasing_epochs += 1
                if self.args.early_stopping > 0 and non_increasing_epochs >= self.args.early_stopping:                
                    logger.info(f'Early stopping, no increase for {non_increasing_epochs} epochs')
                    self.stopped_early = True
                    break
                elif (epoch+1) % checkpoint_interval == 0:
                    filename = f"epoch-{epoch+1}_glblstep-{global_step}_evalloss-{eval_loss}.pt"
                    self.checkpoint_training(filename=filename, optimizer=optimizer, epoch=epoch+1, global_step=global_step, stopped_early=False)
                    self.writer.add_text("ModelCheckpoint", f"file: {filename}\neval_loss: {eval_loss}\nepoch:{epoch+1}", global_step)

            last_step = global_step
            last_epoch = epoch
        
        # End of Training
        self.checkpoint_training("end_of_training_checkpoint.pt", optimizer=optimizer, stopped_early=self.stopped_early, epoch=last_epoch+1, global_step=last_step)
        self.writer.flush()
        self.writer.close()
        return None
        
    def eval(self, global_step):
        logger.info("***** Running evaluation *****")
        repr_layers = self.model_args.get('repr_layers')
        self.model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples = 0
        total_steps = 0
        for step, batch in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc="Eval Iteration"):                
            raw_labels, raw_seqs, tokens, labels = batch
            batch_size = len(raw_labels)          
            tokens = tokens.to(device=self.device, non_blocking=True)
            
            with torch.no_grad():
                out = self.model(tokens, repr_layers=repr_layers, return_contacts=False)
                logits = out["logits"].detach().cpu()
            
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.model.alphabet_size), labels.view(-1), ignore_index=-1)
            eval_loss += loss.mean().item()

            outputs = torch.argmax(logits, axis=-1)
            # tmp_eval_accuracy = torch.sum(outputs == labels[(labels != -1)], axis=-1)
            # eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += batch_size
            total_steps += 1

            if step == 0:
                # Write sample text to tensorboard
                pred_seq = tokens_to_seq(self.alphabet, outputs[0].unsqueeze(dim=0))[0]
                masked_seq = tokens_to_seq(self.alphabet, tokens[0].unsqueeze(dim=0))[0]
                mask = "_".join(masked_seq.split("<mask>")).replace("<cls>", "").replace("<pad>", "")
                ground_truth = raw_seqs[0]
                label = raw_labels[0]
                summary = f"""
                sequence:\t {ground_truth}
                pred_seq:\t {pred_seq[1:]}
                mask:\t\t {mask}
                label:\t {label}
                """
                self.eval_writer.add_text(tag='Text Summary', text_string=summary, global_step=global_step)

        eval_loss /= total_steps
        self.eval_writer.add_scalar("Loss", eval_loss, global_step)
        # eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss} # 'eval_accuracy': eval_accuracy 'global_step': global_step, 'loss': tr_loss/nb_tr_steps}
        self.model.train()

        return result