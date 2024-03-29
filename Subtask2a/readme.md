# Subtask2a
> Code for the [Big Data Analytics](https://bda-hull.github.io/) research group [SemEval2024 Task 4](https://propaganda.math.unipd.it/semeval2024task4/index.html) Subtask2a entry.


## Models
Each file contains the models for Subtask2a. The ensembles can be changed however you would like to test the models, although these files represent the original model structure as in our paper.

To utilise the F1 Hierarchy and Late Fusion Engine, this is done in the Evaluation Code:
- Read the data and the models
- Conduct the evaluation: for the Late Fusion engine, set the output to 'ensemble'. These output files can then be used directly with the late fusion engine. It is advisable not to use the F1 Hierarchy before this, however feel free to experiment!
- Conduct the evaluation with F1 only: as above, but switch from predicted_output to predicted_outputf1h.
- Late Fusion outputs can be read in as .json files and passed through the F1 Hierarchy without running inference on a model. This code will be updated shortly with an example of this.

As a note, in our experiments late fusion is primarily used with the BertExternal model and CLIP. Note that CLIP is trained using Keras/Tensorflow, whereas the other ensembles and Bert(ex) use PyTorch at present.

## Google Vision (BERT(ex))
This is a seperate model which reads in Google Vision entities as text. You can download these entities for the training and test sets [here](https://drive.google.com/drive/folders/14PhBsqzrEa4UjjTITCF8pLPWTa8SW6ek).

## Task Data
Copies of the task data are available from the [task website](https://propaganda.math.unipd.it/semeval2024task4). For copies of the augmented data used for training (e.g., direct translated data), please contact the lead author: v.sherratt-2020@hull.ac.uk. Alternatively, you can recreate the sampled training data using GPT-3.5, albeit the random sample you choose from the task data may differ from ours.

