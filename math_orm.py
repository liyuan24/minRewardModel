"""
format: black --line-length 88 math_orm.py
"""

import torch
import torch.nn as nn
import datasets
from torch.utils.data import Dataset
import json
from training_config import TrainingConfig
from transformers import AutoTokenizer, AutoModel
from typing import Any
from torch.utils.data import DataLoader
from trainer import Trainer

dtype_map = {
    "bfloat16": torch.bfloat16,
}


class DataCollatorForORM:
    """
    The data collator will primarily do three things:
    1. pad the input text to the max length of the batch
    2. set the label values for the non-solution response tokens as ignore_index
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        ignore_index: int,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    @staticmethod
    def get_question(input: str):
        """
        The question is before Step 1 in the input
        """
        return input.split("Step 1")[0]

    def process(self, examples: list[dict[str, Any]]):
        """
        process a batch of examples
        """
        inputs = [example["input"] for example in examples]
        questions = [self.get_question(example["input"]) for example in examples]
        labels = [example["label"] for example in examples]
        tokenized_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        tokenized_questions = self.tokenizer(
            questions, padding=True, return_tensors="pt"
        )
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        question_length = tokenized_questions["attention_mask"].sum(dim=1)
        # Create a label tensor initialized with ignore_index
        label_tensor = torch.full_like(
            input_ids, fill_value=self.ignore_index, dtype=torch.long
        )

        # Set labels for the solution part of labels as label value(1 or 0)
        for i in range(input_ids.size(0)):  # Iterate over batch
            # Set the label for relevant tokens to the sequence label
            label_tensor[i, question_length[i] :] = labels[i]
            # Ensure padding tokens remain as ignore_index
            label_tensor[i, attention_mask[i] == 0] = self.ignore_index
        batch = {
            "input_ids": input_ids,
            "label_tensor": label_tensor,
            "attention_mask": attention_mask,
            "label_values": torch.tensor(labels, dtype=torch.long),
        }
        return batch


class ORMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {}
        sample["input"] = self.dataset[idx][0]["content"]
        sample["label"] = 1 if self.dataset[idx][1]["content"] == "+" else 0
        return sample


class ORMModel(nn.Module):
    def __init__(self, model_name, torch_dtype):
        super(ORMModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.classification_head = nn.Linear(
            self.model.config.hidden_size, 2, dtype=dtype_map[torch_dtype]
        )

    def forward(self, input_ids, attention_mask=None):
        # Get the outputs from the base model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        classification_logits = self.classification_head(last_hidden_state)
        return classification_logits


if __name__ == "__main__":
    orm_training_config_file = "orm.json"
    with open(orm_training_config_file, "r") as f:
        training_config = json.load(f)
    orm_training_config = TrainingConfig(**training_config)
    dataset = datasets.load_dataset(orm_training_config.dataset_name)
    eval_data_size = (
        orm_training_config.validation_data_size + orm_training_config.test_data_size
    )
    train_dataset = dataset["train"]["conversations"][:-eval_data_size]
    val_dataset = dataset["train"]["conversations"][
        -eval_data_size : -orm_training_config.validation_data_size
    ]
    test_dataset = dataset["train"]["conversations"][
        -orm_training_config.validation_data_size :
    ]
    train_dataset = ORMDataset(train_dataset)
    val_dataset = ORMDataset(val_dataset)
    test_dataset = ORMDataset(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(orm_training_config.base_model_name)
    model = ORMModel(orm_training_config.base_model_name, orm_training_config.dtype).to(
        orm_training_config.device
    )
    data_collator = DataCollatorForORM(tokenizer, -100)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=orm_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=orm_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=orm_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        training_config=orm_training_config,
    )
    trainer.train()
