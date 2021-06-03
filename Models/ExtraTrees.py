# Model: Extra forests

import IML_Hackathon2021.CreateDataframe as data
from sklearn.ensemble import ExtraTreesClassifier

# Hyper parameters
ESTIMATOR_NUM = 100
MAX_DEPTH = 20

# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
validation_labels = data.validation_p_labels

# Train
Trees = ExtraTreesClassifier(n_estimators=ESTIMATOR_NUM, max_depth=MAX_DEPTH)
Trees.fit(train_features, train_labels)

# Predict (using validation data)
prediction = Trees.predict(validation_features)
print(prediction)
print(Trees.score(validation_features, validation_labels))
