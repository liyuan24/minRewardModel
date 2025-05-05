from trainer import Trainer
import torch
import torch.nn as nn

class BTRMTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        training_config,
    ):
        super().__init__(model, train_dataloader, eval_dataloader, training_config, nn.BCEWithLogitsLoss())

    def compute_loss(self, criterion, input_ids, attention_mask, labels):
        # [2 * batch_size, seq_len, 1]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # find the last non-padded token and choose the logits of that token
        last_non_padded_token_inds = (attention_mask.sum(dim=1) - 1)
        # [2 * batch_size, 1]
        outputs = outputs[torch.arange(outputs.shape[0]), last_non_padded_token_inds]
        batch_size = outputs.shape[0]
        chose_inds = torch.arange(0, batch_size, 2)
        reject_inds = chose_inds + 1
        chosen_logits = outputs[chose_inds]
        reject_logits = outputs[reject_inds]
        logits = chosen_logits - reject_logits
        # convert the ranking problem into a classification problem
        loss = criterion(logits.view(-1), labels)
        acc_cnt = (chosen_logits > reject_logits).sum().item()
        return loss, acc_cnt
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = []
        total_acc = 0
        total_samples = 0
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            eval_labels = batch["eval_labels"].to(self.device)
            loss, acc_cnt = self.compute_loss(self.criterion, input_ids, attention_mask, eval_labels)
            losses.append(loss.item())
            total_acc += acc_cnt
            total_samples += eval_labels.shape[0]
        self.model.train()
        print(f"eval accuracy: {total_acc / total_samples}")
        return sum(losses) / len(losses)
