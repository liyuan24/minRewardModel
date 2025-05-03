import gzip
from typing import List, Dict, Any
import blobfile as bf
from json import loads as json_loads
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
from torch.utils.data import DataLoader
from math_orm import TrainingConfig
import json
from transformers import AutoModelForCausalLM
from prm_trainer import PRMTrainer

STEP_TEMPLATE = "Step: {step_number}: {step_text}{label_token}\n"
PRM_LABEL_TOKEN = "<|prm_label|>"
POSITIVE_LABEL_TOKEN = "+"
NEGATIVE_LABEL_TOKEN = "-"
dtype_map = {
    "bfloat16": torch.bfloat16,
}


def open_jsonl(file: str):
    if file.endswith(".gz"):
        return gzip.open(bf.BlobFile(file, "rb"))
    return bf.BlobFile(file, "r")


def _read_jsonl(file: str) -> List[Dict]:
    assert bf.exists(file), file
    with open_jsonl(file) as f:
        return [json_loads(l) for l in f.readlines() if l]


def get_label(rating: int) -> int:
    """
    rating -1(incorrect), 0(neutral), 1(correct)
    follow https://arxiv.org/abs/2305.20050 to use neutral as correct
    """
    if rating == 0 or rating == 1:
        return POSITIVE_LABEL_TOKEN
    else:
        return NEGATIVE_LABEL_TOKEN


"""
concatenate problem and reasoning steps and add a special prm label token at the end of each step.
The special token is used to locate the prm label for each step
And we will use the logits for POSITIVE_LABEL_TOKEN and NEGATIVE_LABEL_TOKEN to compute the loss

An example sample is as follows:

The first four terms in an arithmetic sequence are $x+y$, $x-y$, $xy$, and $x/y$, in that order. What is the fifth term? Express your answer as a common fraction.
Step: 1: To find the fifth term, I need to identify the common difference of the arithmetic sequence and add it to the fourth term.<|prm_label|>
Step: 2: The common difference is the same for any consecutive pair of terms, so I can use any of them to find it.<|prm_label|>
Step: 3: For example, using the first and second terms, I can write $x-y = x+y + d$, where $d$ is the common difference.<|prm_label|>
Step: 4: Solving for $d$, I get $d = -2y$.<|prm_label|>
Step: 5: Using another pair of terms, such as the second and third, I can check if this value of $d$ is consistent.<|prm_label|>
Step: 6: I have $xy = x-y + d$, so substituting $d = -2y$, I get $xy = x-y - 2y$.<|prm_label|>
Step: 7: Simplifying, I get $xy = x - 3y$.<|prm_label|>
Step: 8: This seems like a reasonable equation, so I will assume that $d = -2y$ is correct.<|prm_label|>
Step: 9: Now, to find the fifth term, I need to add $d$ to the fourth term.<|prm_label|>
Step: 10: The fourth term is $x/y$, so the fifth term is $x/y + d = x/y - 2y$.<|prm_label|>
Step: 11: To express this as a common fraction, I need to find a common denominator for $x/y$ and $-2y$.<|prm_label|>
Step: 12: The least common denominator is $y$, so I can multiply the numerator and denominator of $-2y$ by $y$ to get $-2y^2/y$.<|prm_label|>
Step: 13: Therefore, the fifth term is $x/y - 2y^2/y = (x - 2y^2)/y$.

# Answer

(x - 2y^2)/y<|prm_label|>
"""


def construct_dataset(data: List[Dict]) -> List[Dict]:
    dataset = []
    for sample in data:
        input = sample["question"]["problem"]
        labels = []
        for i, step in enumerate(sample["label"]["steps"]):
            chosen_completion = (
                step["chosen_completion"]
                if step["chosen_completion"] is not None
                else 0
            )
            # here make a simplification for alternative completions, just choose the first one
            input += STEP_TEMPLATE.format(
                step_number=i + 1,
                step_text=step["completions"][chosen_completion]["text"],
                label_token=PRM_LABEL_TOKEN,
            )
            labels.append(get_label(step["completions"][chosen_completion]["rating"]))
        dataset.append({"input": input, "label": labels})
    return dataset


"""
Filter out samples when finish_reason is *bad_problem* or *give_up*.
Check https://github.com/openai/prm800k/tree/main for more details
"""


def process_data(data: List[Dict]) -> List[Dict]:
    return [
        sample
        for sample in data
        if sample["label"]["finish_reason"] not in ["bad_problem", "give_up"]
    ]


class DataCollatorForPRM:
    """
    The data collator will primarily do three things:
    1. pad the input text to the max length of the batch
    2. convert the PRM labels(+ or -) to their token ids
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
        inputs = [example["input"] for example in examples]
        # flatten the labels
        labels = [label for example in examples for label in example["label"]]
        tokenized_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        label_token_ids = self.tokenizer.convert_tokens_to_ids(labels)
        label_tensor = torch.tensor(label_token_ids, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "training_labels": label_tensor,
            "attention_mask": attention_mask,
            "eval_labels": label_tensor,
        }
        return batch


class PRMDataset(Dataset):
    def __init__(self, dataset: list[dict[str, Any]]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {}
        sample["input"] = self.dataset[idx]["input"]
        sample["label"] = self.dataset[idx]["label"]
        return sample


if __name__ == "__main__":
    prm_training_config_file = "prm.json"
    with open(prm_training_config_file, "r") as f:
        training_config = json.load(f)
    prm_training_config = TrainingConfig(**training_config)

    train_data_path = "prm_data/phase2_train.jsonl"
    test_data_path = "prm_data/phase2_test.jsonl"
    train_data = _read_jsonl(train_data_path)
    test_data = _read_jsonl(test_data_path)
    train_data = process_data(train_data)
    test_data = process_data(test_data)
    train_raw_dataset = construct_dataset(train_data)
    test_raw_dataset = construct_dataset(test_data)
    # Shuffle and split the train_dataset
    seed = 42
    random.seed(seed)
    random.shuffle(train_raw_dataset)

    # Define the split ratio
    split_ratio = 0.97
    split_index = int(len(train_raw_dataset) * split_ratio)

    train_split = train_raw_dataset[:split_index]
    validation_split = train_raw_dataset[split_index:]

    train_dataset = PRMDataset(train_split)
    validation_dataset = PRMDataset(validation_split)
    test_dataset = PRMDataset(test_raw_dataset)

    tokenizer = AutoTokenizer.from_pretrained(prm_training_config.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # add prm label token as a special token
    num_added_toks = tokenizer.add_tokens([PRM_LABEL_TOKEN])
    model = AutoModelForCausalLM.from_pretrained(
        prm_training_config.base_model_name
    ).to(prm_training_config.device)
    # resize the model token embeddings to include the new token
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForPRM(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=prm_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=prm_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=prm_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    trainer = PRMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        training_config=prm_training_config,
        prm_label_token_id=tokenizer.convert_tokens_to_ids(PRM_LABEL_TOKEN),
        positive_label_token_id=tokenizer.convert_tokens_to_ids(POSITIVE_LABEL_TOKEN),
        negative_label_token_id=tokenizer.convert_tokens_to_ids(NEGATIVE_LABEL_TOKEN),
    )
    trainer.train()
