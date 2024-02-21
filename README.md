# Big Data Analytics SemEval2024 Entry

> Code for the [Big Data Analytics](https://bda-hull.github.io/) research group [SemEval2024 Task 4](https://propaganda.math.unipd.it/semeval2024task4/index.html) entry.

This is our entry to [SemEval2024 Task 4: Multilingual Detection of Persuasion Techniques in Memes](https://propaganda.math.unipd.it/semeval2024task4/index.html). Our plan is to tackle only tasks **1** and **2a**.

**Paper:** \[LINK PENDING. TODO FILL IN HERE WHEN AVAILABLE.\]

**This repository is a work in progress.**

## System Requirements
- TODO detail system requirements here

## Getting started
The code in this repository is split into multiple subdirectories:

- **`EDA`:** Exploratory Data Analysis. Extra experiments not required to utilise our approach.
- **`GoogleVision`:** Generates entities from image files. This is used as an input to the vision stream.
- **`LateFusionEngine`:** The late-fusion engine that merges the output of the NLP and Vision streams together using an per-label accuracy weighting system.
- **`Subtask2a`:** Contains the training code, F1 hiersarchy (Evaluation Code) and models used for Subtask2a.
- **`Subtask2b`:** Contains the training code, models and post-evaluation models from Subtask2b.
- **`Test Prediction Files`:** Test prediction files which can be used to test the late fusion engine.
- **`Unimodal Baselines`:** 
- **`scorer-baseline`:** 
- **`word-embeddings`:** Some experiments with word embedding algorithms. These experiments informed the rest of the work done, but is not required to use the approach detailed in our paper.

TODO Fill in the rest of the above descriptions.

Please visit the README.md file in each subdirectory for specific instructions on each subproject.

A common first step though is to clone this git repository:

```bash
git clone https://github.com/vemchance/BDA-SemEval4.git
cd BDA-SemEval4
```

TODO Finish this section of the README. We should include a high-level overview of the project and how to use it.

## Architecture
TODO fill this out.

## OneDrive Link
Link to OneDrive where the big files live: [OneDrive](https://hullacuk-my.sharepoint.com/:f:/g/personal/v_sherratt-2020_hull_ac_uk/EpevevOycPdKppCMZaSyysgB-z2AeAiZ-2YtVN9tHKF-5Q?e=8Of06X)
To be DELETED once the repo is public. E-mail Vic if you don't have access. Do not reshare link.

## Contributing
TODO fill this out. I assume contributions are welcome after the challenge is finished. If so, we should say so here.

## Licence
All code in this repository is licensed under the [GNU Affero General Public License 3.0](./LICENSE.md). choosealicense.com has a great summary of this license: <https://choosealicense.com/licenses/agpl-3.0/>

[AGPL-3.0](https://choosealicense.com/licenses/agpl-3.0/) was chosen as it provides strong copyleft protections to ensure transparency on future and derivative works. Open Science for the win!

TODO insert appropriate meme here
