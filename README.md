This repo contains code for

* Process Reward Model(PRM): for math problem solving
* Outcome Reward Model(ORM): for math problem solving
* Bradley-Terry Model: for chat alignment

For PRM and ORM, the implementation is based on the idea in [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050).

And for Bradley-Terry Model, the implementation is following the idea in [InstructGPT](https://arxiv.org/abs/2203.02155).

# Experiment Setup

I only have 1 GTX 3090 24GB which has very limited memory. So here are a few tricks to run the experiment under this hard constraint

1. Base model: [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) which is a relatively small model
2. Use batch size 1 and gradient accumulation steps 32 to simulate a batch size of 32
3. Use [8-bit Adam optimizer](https://arxiv.org/abs/2110.02861) to reduce the optimizer state memory

# Outcome Reward Model

The outcome reward model uses the correctness of the outcome of the problem as the reward signal to train the model. More specifically, this is a binary classification problem and the
label is *1* if the outcome is correct and *0* otherwise.

## Data Stats
We use [RLHFlow/Mistral-ORM-Data](https://huggingface.co/datasets/RLHFlow/Mistral-ORM-Data) as the training and evaluation data. The statistics of the data is as follows:

| Dataset Split| Min Length(# of tokens) | Max Length(# of tokens) | Average Length(# of tokens) | Median Length(# of tokens) | Correct Answer Count | Incorrect Answer Count | Total Count |
|---------|------------|------------|----------------|---------------|----------------|----------------|-------------|
| Train   | 36        | 2007        | 258.64          | 217.0          | 82314            | 189912            | 272226         |
| Val     | 75        | 1088        | 254.27          | 211.0          | 139            | 361            | 500         |
| Test    | 77        | 1353        | 249.97          | 219.0          | 147            | 353            | 500         |

## Training
Following [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), we use the *per token classification* loss function to train the reward model.

## Evaluation
Still following [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), we use the score from the last token of the reward model as the score of the problem.

# Process Reward Model

Opposed to the outcome reward model, the process reward model will use the correctness of each reasoning step as the reward signal to train the model. This is more fine-grained and more effective. But the shortcoming is that
we need to collect the human labels for each reasoning step.

## Data

We use [PRM800K](https://github.com/openai/prm800k) from OpenAI as the training data.

## Training

As mentioned in [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), the classification problem is turned into a **auto-regressive** problem which is amazing. It effectively uses the unsupervised training pipeline to train a supervised model. Again, excellent idea.

More specifically, there would be a special token at the end of each reasoning step. This special token is used to locate the classification place for each step. The output for this token is a vector of length vocabulary size.
And we will use scores/logits of two tokens(**+/-**) to represent the probability of the correctness of the step.

## Evaluation

The evaluation is pretty much the same as the training.

# Bradley-Terry Model

[Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) is used to model the ranking between two items. In our case, the ranking is based on the human preference. The reward model is used to predict the reward of an LLM response. If response A is preferred over response B, then the reward of A should be higher than that of B.

## Data

We use [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) as the training/evaluation data.

## Training

A chat between user and LLM is the input to a pretrained LLM. The model will output a score and we use it as the reward of the chat. Each sample contains two chats: one is chosen and the other is rejected. So we expect the score of the chosen chat to be higher than that of the rejected one.

## Evaluation

Pretty much the same as the training.