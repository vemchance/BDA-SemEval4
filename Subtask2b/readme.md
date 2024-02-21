# Subtask2a
> Code for the [Big Data Analytics](https://bda-hull.github.io/) research group [SemEval2024 Task 4](https://propaganda.math.unipd.it/semeval2024task4/index.html) Subtask2b entry.


## Models
Each file contains the models for Subtask2b. The ensembles can be changed however you would like to test the models, although these files represent the original model structure as in our paper. We also include our post-evaluation model which improved our results.

- The BERTExternal model is contained within the Subtask2b Ensemble file. This can be trained seperately, or as part of the model. We will update this repository soon with a singular BertExternal model for this task.
- Similarly, VGG16 can be changed to ResNet50 as per experiments. We will also update this repository with these experiments.

As a note, in our experiments late fusion is primarily used with the BertExternal model.

## Google Vision (BERT(ex))
This is a seperate model which reads in Google Vision entities as text. [Contact the lead paper author](v.sherratt-2020@hull.ac.uk) for a copy of these entities.

## Task Data
Copies of the task data are available from the [task website](https://propaganda.math.unipd.it/semeval2024task4). For copies of the augmented data used for training (e.g., direct translated data and Memotion augmented data), please [Contact the lead paper author](v.sherratt-2020@hull.ac.uk).

