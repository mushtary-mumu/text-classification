[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/R1vgPUT1)


### Project Title

## Multi-label Text Classification of Toxic Comments Using Transformer- and Non-transformer-based Models

>#### *OSE Final Project, University of Bonn SS 2023*
>#### Author
>*Mudabbira Mushtary\
>Student of M.Sc. Economics\
>University of Bonn*

### Questions and Answers

The answers to the [questions](https://dl-intro.readthedocs.io/en/latest/questions.html) from the course are available here: [OSE: Answers to Questions](OSE_2023_Mushtary_Q_and_A.pdf)

### Description

Aim of this project is to classify the comments collected from [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) available on [Kaggle](https://www.kaggle.com/).

The comments are classified into six categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.

Detailed information on the dataset used, the methods applied and the results obtained are available here :arrow_right: [Methods and Results](PROJECT_INFO.md)

### How to run the project:

The following steps are to be followed to run the project:

1. Open a new notebook on Google Colab

2. Mount the google drive to the notebook:

```
from google.colab import drive
drive.mount('/content/drive')
```

3. Clone the repository:
```
username = "iame-uni-bonn"
repository = "final-project-mushtary-mumu"
!git clone https://{token}@github.com/{username}/{repository}

```
`token` is the personal access token generated from github. The steps to generate the token are given available here: https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token

4. Navigate to the repository and open the notebook titled `multilabel_classification.ipynb`

5. Once the notebook is open, change the runtime to GPU by going to `Runtime` -> `Change runtime type` -> `Hardware accelerator` -> `GPU` -> `Save`

6. Run the notebook




