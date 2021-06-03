# Model: Neural Network

import IML_Hackathon2021.CreateDataframe as data
from sklearn.neural_network import MLPClassifier

# Hyper parameters
HIDDEN_LAYERS = 100

# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
validation_labels = data.validation_p_labels

# Train
MLPClassifier= MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS)
MLPClassifier.fit(train_features, train_labels)

# Predict (using validation data)
prediction = MLPClassifier.predict(validation_features)
print(prediction)
print(MLPClassifier.score(validation_features,validation_labels))