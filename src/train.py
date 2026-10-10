from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow

mlflow.set_experiment("Iris_MLOps")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

for n_estimators in [50, 100, 200]:

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)

        print("n_estimators :", n_estimators)
        print("Accuracy :", accuracy)

        os.makedirs("models", exist_ok=True)
        joblib.dump(
            model,
            f"models/iris_model_{n_estimators}.pkl"
        )