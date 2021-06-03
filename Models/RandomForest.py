# Model: Random forest

from sklearn.ensemble import RandomForestClassifier
import IML_Hackathon2021.CreateDataframe as data

# Hyper parameters
ESTIMATOR_NUM = 100
MAX_DEPTH = 20

# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
validation_labels = data.validation_p_labels

# Train
RandomForest= RandomForestClassifier(n_estimators=ESTIMATOR_NUM, max_depth=MAX_DEPTH)
RandomForest.fit(train_features, train_labels)

# Predict (using validation data)
prediction = RandomForest.predict(validation_features)
print(prediction)
print(RandomForest.score(validation_features,validation_labels))