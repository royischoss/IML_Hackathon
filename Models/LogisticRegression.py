# Model: Logistic Regression

import IML_Hackathon2021.CreateDataframe as data
from sklearn.linear_model import LogisticRegression

# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
validation_labels = data.validation_p_labels

# Train
Logistic = LogisticRegression()
Logistic.fit(train_features, train_labels)

# Predict (using validation data)
prediction = Logistic.predict(validation_features)
print(prediction)
print(Logistic.score(validation_features,validation_labels))