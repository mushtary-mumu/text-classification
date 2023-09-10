### Description

This project approaches the task of multi-labeltext classification from two different perspectives. The first approach is based on the traditional machine learning techniques and the second approach is based on deep learning techniques.

The traditional machine learning techniques used are:
- Logistic Regression
- Naive Bayes

The deep learning techniques used are:
- Fine tuning a pretrained BERT model and then using it for classification

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).

The dataset contains 159571 comments which are classified into six categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.

The dataset is highly imbalanced. The following table shows the number of comments in each category.

| Category | Number of comments |
| --- | --- |
| Toxic | 15294 |
| Severe Toxic | 1595 |
| Obscene | 8449 |
| Threat | 478 |
| Insult | 7877 |
| Identity Hate | 1405 |

The dataset is split into train and test sets. The train set contains 159571 comments and the test set contains 63978 comments.

Pytorch lightning is used for training the models since it provides a high level interface for Pytorch. It also provides a lot of useful features like automatic checkpointing, automatic logging, automatic gradient accumulation, etc.

The models are trained on Google Colab. MEASURED TRAINING RUNTIME: 1h 8m 31s

### Results

The following table shows the results obtained on the test set. The results are obtained by averaging the results obtained on each category. The results are obtained by using the F1 score as the metric.

| Model | F1 score |
| --- | --- |
| Logistic Regression | 0.755 |
| Naive Bayes | 0.755 |
| BERT | 0.866 |

The results obtained by using the BERT model are better than the results obtained by using the traditional machine learning techniques. This is because the BERT model is able to capture the context of the comments and hence is able to classify the comments better.

### References

- [BERT](https://arxiv.org/abs/1810.04805)
- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

### Future Work

- Use other deep learning models like CNN, LSTM, etc.
- Use other pretrained models like XLNet, etc.
- Use other machine learning models like SVM, etc.
- Use other machine learning techniques like Word2Vec, etc.
- Use other metrics like precision, recall, etc.
- Use other datasets like [Wikipedia Talk Labels: Toxicity](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Toxicity/4563973), [Wikipedia Talk Labels: Personal Attacks](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689), etc.


