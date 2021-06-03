from sklearn.neighbors import KNeighborsClassifier
import CreateDataframe

# Hyper parameters
KNN_REG_PARAM = 5


# Train
KNN = KNeighborsClassifier(n_neighbors=KNN_REG_PARAM)
train_features = CreateDataframe.train_p_features
train_labels = CreateDataframe.train_p_labels
validation_features = CreateDataframe.validation_p_features
KNN.fit(train_features, train_labels)

#Predict
KNN.predict(validation_features)
