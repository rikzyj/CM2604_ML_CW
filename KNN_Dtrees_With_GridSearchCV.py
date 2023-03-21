import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and preprocess the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
column_names = [f"attribute_{i}" for i in range(1, 58)] + ["label"]
data = pd.read_csv(url, header=None, names=column_names)

# Split the dataset into training and testing sets
X = data.iloc[:, :-1]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Model
knn_params = {"n_neighbors": np.arange(1, 21), "weights": ["uniform", "distance"]}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring="accuracy")
knn_grid.fit(X_train, y_train)
knn_best = knn_grid.best_estimator_

# Decision Tree Model
tree_params = {"criterion": ["gini", "entropy"], "max_depth": np.arange(1, 21)}
tree = DecisionTreeClassifier()
tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring="accuracy")
tree_grid.fit(X_train, y_train)
tree_best = tree_grid.best_estimator_

# Model Evaluation
models = {"KNN": knn_best, "Decision Tree": tree_best}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
