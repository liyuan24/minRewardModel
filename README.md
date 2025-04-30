# Data Stats

## Outcome Reward Model
We use [RLHFlow/Mistral-ORM-Data](https://huggingface.co/datasets/RLHFlow/Mistral-ORM-Data) as the training and evaluation data. The statistics of the data is as follows:

| Dataset Split| Min Length(# of tokens) | Max Length(# of tokens) | Average Length(# of tokens) | Median Length(# of tokens) | Correct Answer Count | Incorrect Answer Count | Total Count |
|---------|------------|------------|----------------|---------------|----------------|----------------|-------------|
| Train   | 36        | 2007        | 258.64          | 217.0          | 82314            | 189912            | 272226         |
| Val     | 75        | 1088        | 254.27          | 211.0          | 139            | 361            | 500         |
| Test    | 77        | 1353        | 249.97          | 219.0          | 147            | 353            | 500         |
