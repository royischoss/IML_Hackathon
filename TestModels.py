import CreateDataframe as data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Hyper parameters
BOOSTER_ESTIMATOR_NUM = 200
TREES_ESTIMATOR_NUM = 100
TREES_MAX_DEPTH = 20
FOREST_ESTIMATOR_NUM = 100
FOREST_MAX_DEPTH = 20
K = 173
HIDDEN_LAYERS = 100


# data
train_features = data.train_p_features
train_labels = data.train_p_labels
validation_features = data.validation_p_features
validation_labels = data.validation_p_labels

#create models
booster = AdaBoostClassifier(n_estimators=BOOSTER_ESTIMATOR_NUM)
trees = ExtraTreesClassifier(n_estimators=TREES_ESTIMATOR_NUM, max_depth=TREES_MAX_DEPTH)
forest= RandomForestClassifier(n_estimators=FOREST_ESTIMATOR_NUM, max_depth=FOREST_MAX_DEPTH)
KNN = KNeighborsClassifier(n_neighbors=K)
logistic = LogisticRegression()
MLP = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS)

models = [booster,trees,forest,KNN,logistic,MLP]
models_names = ["Adaboost","Extra Trees","Random Forest","KNN","Logistic","MLP"]

#train models
for model in models:
    model.fit(train_features,train_labels)

# print predicted accuracy on validation data
for i,model in enumerate(models):
    accuracy = "{:.3f}".format(model.score(validation_features, validation_labels))
    print("Accuracy for " + models_names[i] + ": " + str(accuracy))


