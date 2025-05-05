import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from typing import Any
import json
from training_config import TrainingConfig
import datasets
from torch.utils.data import DataLoader
from bradley_terry_trainer import BTRMTrainer

dtype_map = {
    "bfloat16": torch.bfloat16,
}


class DataCollator:
    """
    pad the input text to the max length of the batch and make chosen and rejected as separate input
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
    ):
        self.tokenizer = tokenizer

    def process(self, examples: list[dict[str, Any]]):
        """
        process a batch of examples
        """
        chosens = [example["chosen"] for example in examples]
        rejs = [example["rejected"] for example in examples]
        # interleave chosens and rejs as separate inputs
        inputs = []
        for i in range(len(chosens)):
            inputs.append(chosens[i])
            inputs.append(rejs[i])
        tokenized_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # the labels are all 1 because we will use log(sigmoid(chosen_logits - rejected_logits)) as the loss function
            "training_labels": torch.ones(len(chosens)),
            "eval_labels": torch.ones(len(chosens)),
        }
        return batch


class BTRMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {}
        chosen = self.dataset[idx]["chosen"]
        rejected = self.dataset[idx]["rejected"]
        sample["chosen"] = chosen
        sample["rejected"] = rejected
        return sample


class BTRewardModel(nn.Module):
    def __init__(self, model_name, torch_dtype):
        super(BTRewardModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.classification_head = nn.Linear(
            self.model.config.hidden_size, 1, dtype=dtype_map[torch_dtype]
        )

    def forward(self, input_ids, attention_mask=None):
        # Get the outputs from the base model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        classification_logits = self.classification_head(last_hidden_state)
        return classification_logits


if __name__ == "__main__":
    training_config_file = "bradley_terry.json"
    with open(training_config_file, "r") as f:
        training_config = json.load(f)
    training_config = TrainingConfig(**training_config)
    dataset = datasets.load_dataset(training_config.dataset_name)
    eval_data_size = training_config.validation_data_size
    train_test_split = dataset['train'].train_test_split(test_size=eval_data_size, seed=42)
    train_dataset = train_test_split["train"]
    validation_dataset = train_test_split["test"]
    test_dataset = dataset["test"]
    train_dataset = BTRMDataset(train_dataset)
    validation_dataset = BTRMDataset(validation_dataset)
    test_dataset = BTRMDataset(test_dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(training_config.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = BTRewardModel(training_config.base_model_name, training_config.dtype).to(
        training_config.device
    )
    data_collator = DataCollator(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )
    
    trainer = BTRMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        training_config=training_config,
    )
    trainer.train()
