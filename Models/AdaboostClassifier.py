# Model: Extra forests

import IML_Hackathon2021.CreateDataframe as data
from sklearn.ensemble import AdaBoostClassifier

# Hyper parameters
ESTIMATOR_NUM = 200

# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
validation_labels = data.validation_p_labels

# Train
booster = AdaBoostClassifier(n_estimators=ESTIMATOR_NUM)
booster.fit(train_features, train_labels)

# Predict (using validation data)
prediction = booster.predict(validation_features)
print(prediction)
print(booster.score(validation_features, validation_labels))
