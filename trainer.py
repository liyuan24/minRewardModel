import torch
from contextlib import nullcontext
import os
from training_config import TrainingConfig
from torch.utils.data import DataLoader
import json
import bitsandbytes as bnb
import math
import torch.nn as nn

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def get_wandb_config(training_config: TrainingConfig):
    config = {
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.learning_rate,
        "use_fused_adamw": training_config.adamw_use_fused,
    }
    return config


def configure_optimizers(
    model, weight_decay, learning_rate, betas, fused, use_eight_bit_optimizer=False
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwis    no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    if use_eight_bit_optimizer:
        # fuse is not supported
        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=betas)
    else:
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=fused
        )
    return optimizer


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        training_config: TrainingConfig,
        criterion: nn.Module = None,
    ):
        global ptdtype
        self.model = model
        self.training_config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ctx = (
            nullcontext()
            if training_config.device == "cpu"
            # mixed precision training
            else torch.amp.autocast(
                device_type=training_config.device, dtype=ptdtype[training_config.dtype]
            )
        )
        self.device = training_config.device
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = []
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            eval_labels = batch["eval_labels"].to(self.device)
            # [batch_size, seq_len, 2]
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # since some sequences in the batch are padded, we use this method to get the output for the last non-padded token
            # for each sequence in this batch
            last_non_padded_indices = attention_mask.sum(dim=1) - 1
            # [batch_size, 2]
            last_outputs = outputs[torch.arange(outputs.size(0)), last_non_padded_indices]
            # this loss is different from the loss in the training loop
            # because we want to use the probability of the last non-padded token as the prediction
            # see https://arxiv.org/abs/2305.20050 for more details
            loss = self.criterion(last_outputs, eval_labels)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)
    
    def compute_loss(self, criterion, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.view(-1, 2), labels.view(-1))
        return loss, None

    def train(self):
        if self.training_config.wandb_log:
            import wandb

            wandb.init(
                project=self.training_config.wandb_project,
                name=self.training_config.wandb_run_name,
                config=get_wandb_config(self.training_config),
            )
        optimizer = configure_optimizers(
            self.model,
            self.training_config.adamw_weight_decay,
            self.training_config.learning_rate,
            (self.training_config.adamw_beta1, self.training_config.adamw_beta2),
            self.training_config.adamw_use_fused,
            self.training_config.use_eight_bit_optimizer,
        )
        optimizer.zero_grad(set_to_none=True)
        best_val_loss = 1e9
        epoch = 0
        iter_num = 0  # each iteration is one batch gradient accumulation
        last_eval_iter = 0
        if self.training_config.resume:
            checkpoint = torch.load(
                os.path.join(self.training_config.out_dir, self.training_config.resume_checkpoint_path)
            )
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_val_loss = checkpoint["best_val_loss"]
            epoch = checkpoint["epoch_num"]
            # iter_num = checkpoint["iter_num"]
            # last_eval_iter = iter_num
        while epoch < self.training_config.epochs:
            total_loss = 0
            optimizer.zero_grad()  # Zero the gradients at the start of each epoch

            for i, batch in enumerate(self.train_dataloader):
                # determine and set the learning rate for this iteration
                lr = (
                    get_lr(
                        iter_num,
                        self.training_config.warmup_iters,
                        self.training_config.lr_decay_iters,
                        self.training_config.learning_rate,
                        self.training_config.min_learning_rate,
                    )
                    if self.training_config.decay_lr
                    else self.training_config.learning_rate
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                input_ids = batch["input_ids"].to(self.training_config.device)
                attention_mask = batch["attention_mask"].to(self.training_config.device)
                training_labels = batch["training_labels"].to(self.training_config.device)

                with self.ctx:
                    # Forward pass
                    loss, _ = self.compute_loss(self.criterion, input_ids, attention_mask, training_labels)

                    # Normalize loss to account for gradient accumulation
                    loss = loss / self.training_config.gradient_accumulation_steps
                    loss.backward()  # Backward pass

                total_loss += loss.item()

                # Update weights and zero gradients every accumulation_steps
                if (i + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.training_config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.training_config.grad_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    iter_num += 1
                if (iter_num + 1) % self.training_config.eval_interval == 0 and iter_num > last_eval_iter:
                    last_eval_iter = iter_num
                    eval_loss = self.evaluate()
                    if self.training_config.wandb_log:
                        wandb.log(
                            {
                                "Step": iter_num,
                                "Train Loss": total_loss / (i + 1),
                                "Val Loss": eval_loss,
                                "Learning Rate": lr,
                            }
                        )
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        if iter_num > 0:
                            checkpoint = {
                                "model": self.model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "iter_num": iter_num,
                                "epoch_num": epoch,
                                "best_val_loss": best_val_loss,
                            }
                            if not os.path.exists(self.training_config.out_dir):
                                os.makedirs(self.training_config.out_dir)
                            torch.save(
                                checkpoint,
                                os.path.join(
                                    self.training_config.out_dir,
                                    f"{self.training_config.checkpoint_path_prefix}_{epoch}_{iter_num}.pt",
                                ),
                            )
                    print(
                        f"epoch {epoch}, step {iter_num+1}, train loss: {total_loss / (i + 1):.4f}, val loss: {eval_loss:.4f}"
                    )
            epoch += 1