# Unimodal Baselines

Jupyter Notebook file containing code to generate unimodal baselines fine-tuned on the subtask 1 dataset only. Note this is only for testing purposes; code is modified version of a SemEval 3 task and is for testing purposes only. Classes and functions can largely be reused to test and fine-tune on larger datasets.


## Data
Skip the first section of the notebook to read in the data, unless you want to modify and use your local files to fine-tune on different data. The actual CSV files used are available in testdata. Note the code currently requires the link (and subsuquent fields) are removed.

## Model Files
Model files are too large to add to repository, stored on OneDrive link (PDF on main) under Unimodal baselines.

## Baseline Results
Tested on the same task (Subtask1a). These results are not expected to be good as the models have only been finetuned on the task data and if the task were this easy... it would not be a competition:

| Model  | Language | Accuracy | F1 Micro | F1 Macro |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BERTBase | English | 0.21  | 0.29  | 0.10  |
| MBert  | English  | 0.21  | 0.28  | 0.11  |
| Roberta Base  | English | 0.24  | 0.40  | 0.17  |
| Roberta Large  | English  | 0.25  | 0.48  | 0.25  |
| XLM Roberta Base | English | 0.20  | 0.17  | 0.007  |
| XLM Roberta Large  | English  | 0.26  | 0.49 | 0.19  |

### TO DO:
Requirements.txt file is missing, but you should be able to run the code and pip install from the imports.
Need to test on multilingual data.

Baselines not been tested - believe BERT uncased achieved around 62% accuracy on the validation set (not bad but small set)
