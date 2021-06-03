# Model: K nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
import IML_Hackathon2021.CreateDataframe as data

# Hyper parameters
KNN_REG_PARAM = 200

# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
validation_labels = data.validation_p_labels

# Train
KNN = KNeighborsClassifier(n_neighbors=KNN_REG_PARAM)
KNN.fit(train_features, train_labels)

# Predict
prediction = KNN.predict(validation_features)
print(prediction)
print(KNN.score(validation_features,validation_labels))
