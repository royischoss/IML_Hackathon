# Model: K nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
import IML_Hackathon2021.CreateDataframe as data

# Hyper parameters
KNN_REG_PARAM = 5

# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
# print("train features:\n")
# print(train_features.head())
# print("\ntrain labels:\n")
# print(train_labels.head())
# print("\nvalidation features:\n")
# print(validation_features.head())

# Train
KNN = KNeighborsClassifier(n_neighbors=KNN_REG_PARAM)
KNN.fit(train_features, train_labels)

# Predict
KNN.predict(validation_features)
