# Experiment Setup

I only have 1 GTX 3090 24GB which has very limited memory. So here are a few tricks to run the experiment under this hard constraint

1. Base model: [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) which is a relatively small model
2. Use batch size 1 and gradient accumulation steps 32 to simulate a batch size of 32
3. Use [8-bit Adam optimizer](https://arxiv.org/abs/2110.02861) to reduce the optimizer state memory

# Outcome Reward Model

## Data Stats
We use [RLHFlow/Mistral-ORM-Data](https://huggingface.co/datasets/RLHFlow/Mistral-ORM-Data) as the training and evaluation data. The statistics of the data is as follows:

| Dataset Split| Min Length(# of tokens) | Max Length(# of tokens) | Average Length(# of tokens) | Median Length(# of tokens) | Correct Answer Count | Incorrect Answer Count | Total Count |
|---------|------------|------------|----------------|---------------|----------------|----------------|-------------|
| Train   | 36        | 2007        | 258.64          | 217.0          | 82314            | 189912            | 272226         |
| Val     | 75        | 1088        | 254.27          | 211.0          | 139            | 361            | 500         |
| Test    | 77        | 1353        | 249.97          | 219.0          | 147            | 353            | 500         |
