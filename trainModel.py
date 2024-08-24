import handDataCollector as hdc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Collect data
finger_data = hdc.get_finger_data()

X = np.array([item[0] for item in finger_data])
y = np.array([item[1] for item in finger_data])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracy_matrix = []

# SVM Model
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
svm_model.fit(X_train, y_train)
svm_accuracy = svm_model.score(X_test, y_test)
print("Model accuracy SVC:", svm_accuracy)
accuracy_matrix.append(svm_accuracy)

# Neural Network Model
nn_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
nn_model.fit(X_train, y_train)
nn_accuracy = nn_model.score(X_test, y_test)
print("Model accuracy Neural Network:", nn_accuracy)
accuracy_matrix.append(nn_accuracy)

# Random Forest Model
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print("Model accuracy Random Forest:", rf_accuracy)
accuracy_matrix.append(rf_accuracy)

# K-Nearest Neighbors Model
knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)
print("Model accuracy KNN:", knn_accuracy)
accuracy_matrix.append(knn_accuracy)

# Logistic Regression Model
lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)
print("Model accuracy Logistic Regression:", lr_accuracy)
accuracy_matrix.append(lr_accuracy)

# Gradient Boosting Model
gb_model = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=100, random_state=42))
gb_model.fit(X_train, y_train)
gb_accuracy = gb_model.score(X_test, y_test)
print("Model accuracy Gradient Boosting:", gb_accuracy)
accuracy_matrix.append(gb_accuracy)

# Naive Bayes Model
nb_model = make_pipeline(StandardScaler(), GaussianNB())
nb_model.fit(X_train, y_train)
nb_accuracy = nb_model.score(X_test, y_test)
print("Model accuracy Naive Bayes:", nb_accuracy)
accuracy_matrix.append(nb_accuracy)

print("Accuracy Matrix:", accuracy_matrix)


# Save the best model
best_model = max(accuracy_matrix)
if best_model == svm_accuracy:
    model = svm_model
elif best_model == nn_accuracy:
    model = nn_model
elif best_model == rf_accuracy:
    model = rf_model
elif best_model == knn_accuracy:
    model = knn_model
elif best_model == lr_accuracy:
    model = lr_model
elif best_model == gb_accuracy:
    model = gb_model
else:
    model = nb_model

import joblib
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")

# # Load the model
# model = joblib.load('model.pkl')

