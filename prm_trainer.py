from trainer import Trainer
import torch
import torch.nn as nn
class PRMTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        training_config,
        prm_label_token_id: int,
        positive_label_token_id: int,
        negative_label_token_id: int,
    ):
        super().__init__(model, train_dataloader, eval_dataloader, training_config)
        self.prm_label_token_id = prm_label_token_id
        self.prm_reward_token_ids = [positive_label_token_id, negative_label_token_id]

    def compute_loss(self, criterion, input_ids, attention_mask, labels):
        # convert labels from token id to 0 and 1
        labels = torch.where(labels == self.prm_reward_token_ids[0], 0, 1)
        # [batch_size, seq_len, vocab_size]
        outputs = self.model(input_ids, attention_mask=attention_mask).logits
        # locate the prm label token
        prm_labels = torch.where(input_ids == self.prm_label_token_id)
        batch_inds = prm_labels[0]
        seq_inds = prm_labels[1]
        vocab_size = outputs.shape[2]
        prm_label_output_logits = outputs[batch_inds, seq_inds]
        # [batch_size * label_seq_len, vocab_size]
        prm_label_output_logits = prm_label_output_logits.view(-1, vocab_size)
        # [batch_size * label_seq_len, 2]
        label_logits = prm_label_output_logits[:, self.prm_reward_token_ids]
        # calculate the loss only on the positive and negative label token ids
        loss = criterion(label_logits, labels)
        return loss, None
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            eval_labels = batch["eval_labels"].to(self.device)
            loss = self.compute_loss(loss_fn, input_ids, attention_mask, eval_labels)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)
