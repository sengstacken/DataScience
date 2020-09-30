# DataScience and Machine Learning Metrics
This page covers metrics that can be used for machine learning

### Regression
- [Machine Learning Jupyter Notebook Template](https://github.com/sengstacken/DataScience/blob/master/machine_learning_template.ipynb)

### Classification

#### Balanced Datasets
- Accuracy (Acc) - Defines how accurate your model is.  Suitable for balanced datasets - sklearn.metrics.accuracy_score(y_true, y_pred

#### Imbalanced Datasets
- Precision (P) - Better for imbalanced datasets.  What percentage of time is the model correct when trying to identify positive examples. TP/(TP+FP) - 

    ```sklearn.metrics.precision_score(y_true, y_pred)```

- Recall (R) - How many of the positive samples were classified correctly TP/(TP+FN) sklearn.metrics.recall_score(y_true, y_pred
- F1 score (F1)
- Area Under Receiver Operating Characteristic (ROC) curve (AUC)
- Log Loss
- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)

