from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    classification_report
)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import json
import os
import shutil
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")  # Sauvegarde du graphique sans ouvrir de fenêtre
import matplotlib.pyplot as plt

# 1. Configuration MLflow
# Le Model Registry a besoin d'une base de données : on utilise SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_MLOps")

MODEL_NAME = "Iris_Model"
MODEL_ALIAS = "champion"

# 2. Chargement des données
iris = load_iris()
X = iris.data
y = iris.target

# 3. Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Création des dossiers
os.makedirs("models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

# Résultats de chaque run
resultats = []

# 5. Entraînement des 3 modèles
for n_estimators in [50, 100, 200]:

    with mlflow.start_run(run_name=f"RF_{n_estimators}") as run:

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Prédictions
        predictions = model.predict(X_test)

        # 6. Évaluation
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        # 7. Tracking MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", 0.2)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)

        # 8. Sauvegarde locale
        model_path = f"models/iris_model_{n_estimators}.pkl"
        joblib.dump(model, model_path)

        # 9. Sauvegarde du modèle dans MLflow
        # Correction de l'erreur skops
        model_info = mlflow.sklearn.log_model(
            model,
            name=f"model_{n_estimators}",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # 10. Affichage des résultats
        print("\n----------------------------")
        print("n_estimators :", n_estimators)
        print("Accuracy :", accuracy)
        print("Precision :", precision)

        print("\nRapport de classification :")
        print(classification_report(
            y_test,
            predictions,
            target_names=iris.target_names
        ))

        print("Modèle sauvegardé :", model_path)

        # Données pour la comparaison
        resultats.append({
            "n_estimators": n_estimators,
            "accuracy": accuracy,
            "precision": precision,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri
        })

# 11. Sélection du meilleur modèle
# En cas d'égalité, on garde le modèle le plus simple (le moins d'arbres)
meilleur = max(resultats, key=lambda r: (r["accuracy"], -r["n_estimators"]))

print("\nMeilleur modèle : n_estimators =", meilleur["n_estimators"])

# 12. Enregistrement dans le Model Registry
version = mlflow.register_model(meilleur["model_uri"], MODEL_NAME)

client = MlflowClient()
client.update_registered_model(
    MODEL_NAME,
    description="Random Forest de classification des fleurs Iris."
)
client.update_model_version(
    MODEL_NAME,
    version.version,
    description=f"Meilleur modèle du run RF_{meilleur['n_estimators']}."
)
client.set_model_version_tag(
    MODEL_NAME, version.version, "accuracy", str(meilleur["accuracy"])
)
client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)

print(f"Registry : {MODEL_NAME} version {version.version} @{MODEL_ALIAS}")

# 13. Modèle utilisé par predict.py et par l'API
shutil.copy(
    f"models/iris_model_{meilleur['n_estimators']}.pkl",
    "models/iris_model.pkl"
)

with open("models/model_info.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "name": MODEL_NAME,
            "version": version.version,
            "n_estimators": meilleur["n_estimators"],
            "accuracy": meilleur["accuracy"],
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
        },
        f,
        indent=2
    )

# 14. Traçage graphique
nb_arbres = [r["n_estimators"] for r in resultats]

plt.figure(figsize=(8, 5))

plt.plot(
    nb_arbres,
    [r["accuracy"] for r in resultats],
    marker="o",
    label="Accuracy"
)

plt.plot(
    nb_arbres,
    [r["precision"] for r in resultats],
    marker="s",
    label="Precision"
)

plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("Score")
plt.title("Comparaison des modèles Random Forest")
plt.xticks(nb_arbres)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)

# 15. Sauvegarde du graphique
graph_path = "experiments/comparaison_modeles.png"

plt.savefig(
    graph_path,
    dpi=150,
    bbox_inches="tight"
)

# 16. Enregistrement du graphique dans le run du meilleur modèle
with mlflow.start_run(run_id=meilleur["run_id"]):
    mlflow.log_artifact(graph_path)

print("\nToutes les expériences sont terminées !")
