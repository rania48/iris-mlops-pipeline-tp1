import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 1. Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

MODEL_NAME = "IrisClassifier"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 2. Dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# 3. Configurations à comparer
experiments = [
    {
        "nom": "RF_50_depth3",
        "estimateur": RandomForestClassifier,
        "params": {"n_estimators": 50, "max_depth": 3, "random_state": RANDOM_STATE}
    },
    {
        "nom": "RF_100_depth5",
        "estimateur": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE}
    },
    # Exercice : min_samples_split et min_samples_leaf
    {
        "nom": "RF_100_min_samples",
        "estimateur": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "random_state": RANDOM_STATE
        }
    },
    # Exercice : LogisticRegression et DecisionTreeClassifier
    {
        "nom": "LogisticRegression",
        "estimateur": LogisticRegression,
        "params": {"max_iter": 1000, "random_state": RANDOM_STATE}
    },
    {
        "nom": "DecisionTree_depth3",
        "estimateur": DecisionTreeClassifier,
        "params": {"max_depth": 3, "random_state": RANDOM_STATE}
    }
]

run_ids = []

# 4. Entraînement et suivi des runs
for config in experiments:

    with mlflow.start_run(run_name=config["nom"]) as run:

        model = config["estimateur"](**config["params"])
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "precision": precision_score(y_test, y_test_pred, average="weighted"),
            "recall": recall_score(y_test, y_test_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
            # Exercice : un grand écart peut indiquer de l'overfitting
            "ecart_train_test": train_accuracy - test_accuracy
        }

        mlflow.log_params({
            "estimator": config["estimateur"].__name__,
            "test_size": TEST_SIZE,
            "dataset": "Iris",
            **config["params"]
        })
        mlflow.log_metrics(metrics)

        # Enregistrer le modèle comme artefact du run
        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        run_ids.append(run.info.run_id)

        print("Run ID :", run.info.run_id)
        print("Nom :", config["nom"])
        print("Métriques :", metrics)

# 5. Classer les runs de cette exécution
# Exercice : sélection selon f1_score plutôt que test_accuracy
runs = mlflow.search_runs(
    experiment_names=["TP_Iris_Classification"],
    order_by=["metrics.f1_score DESC", "metrics.ecart_train_test ASC"]
)
runs = runs[runs["run_id"].isin(run_ids)]

tableau = runs[[
    "tags.mlflow.runName",
    "metrics.train_accuracy",
    "metrics.test_accuracy",
    "metrics.precision",
    "metrics.recall",
    "metrics.f1_score",
    "metrics.ecart_train_test"
]].round(4)

print("\nComparaison des runs :")
print(tableau.to_string(index=False))

os.makedirs("experiments", exist_ok=True)
tableau.to_csv("experiments/comparaison_runs_iris.csv", index=False)

# 6. Enregistrer les 2 meilleurs modèles dans le Model Registry
# Exercice : le meilleur est enregistré en dernier, c'est la dernière version
registered = None

for run_id in reversed(runs["run_id"].head(2).tolist()):
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/iris_model",
        name=MODEL_NAME
    )
    print("Modèle :", registered.name, "| Version :", registered.version)

client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)

# 7. Charger la version enregistrée et prédire
version_uri = f"models:/{MODEL_NAME}/{registered.version}"
loaded_model = mlflow.sklearn.load_model(version_uri)

prediction = loaded_model.predict([X_test[0]])

print("Classe prédite :", iris.target_names[prediction[0]])
print("Classe réelle :", iris.target_names[y_test[0]])
