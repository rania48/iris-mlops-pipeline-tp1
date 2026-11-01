import joblib
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Alerte si la moyenne bouge de plus de 0,5 écart-type
SEUIL_DRIFT = 0.5


def simuler_nouvelles_donnees(reference):
    """Simule des données de production : les pétales deviennent plus grands."""
    generateur = np.random.default_rng(42)

    nouvelles = reference.copy()
    nouvelles["petal length (cm)"] *= 1.3
    nouvelles["petal width (cm)"] *= 1.3
    nouvelles += generateur.normal(0, 0.05, nouvelles.shape)

    return nouvelles.round(2)


def detecter_drift(reference, nouvelles, seuil=SEUIL_DRIFT):
    """Compare moyenne, écart-type, minimum et maximum de chaque variable."""
    rapport = pd.DataFrame({
        "moyenne_ref": reference.mean(),
        "moyenne_new": nouvelles.mean(),
        "std_ref": reference.std(),
        "std_new": nouvelles.std(),
        "min_ref": reference.min(),
        "min_new": nouvelles.min(),
        "max_ref": reference.max(),
        "max_new": nouvelles.max()
    })

    ecart = (rapport["moyenne_new"] - rapport["moyenne_ref"]).abs()
    rapport["score_drift"] = ecart / rapport["std_ref"]
    rapport["alerte"] = rapport["score_drift"] > seuil

    return rapport


def repartition_classes(model, donnees, noms):
    """Pourcentage de fleurs prédites dans chaque classe."""
    predictions = pd.Series(model.predict(donnees.values)).map(lambda c: noms[c])
    repartition = predictions.value_counts(normalize=True) * 100

    return repartition.reindex(noms, fill_value=0).round(1)


if __name__ == "__main__":
    iris = load_iris()

    reference = pd.DataFrame(iris.data, columns=iris.feature_names)
    nouvelles = simuler_nouvelles_donnees(reference)

    os.makedirs("data", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    nouvelles.to_csv("data/nouvelles_donnees.csv", index=False)

    rapport = detecter_drift(reference, nouvelles)
    rapport.round(3).to_csv("experiments/rapport_drift.csv")

    print(rapport[["moyenne_ref", "moyenne_new", "score_drift", "alerte"]].round(3))

    variables = rapport.index[rapport["alerte"]].tolist()

    if variables:
        print("\nALERTE DRIFT sur :", ", ".join(variables))
        print("Action : vérifier les données puis réentraîner le modèle.")
    else:
        print("\nPas de drift détecté.")

    # Impact du drift sur les prédictions
    model = joblib.load("models/iris_model.pkl")

    print("\nClasses prédites (%) :")
    print(pd.DataFrame({
        "reference": repartition_classes(model, reference, iris.target_names),
        "nouvelles": repartition_classes(model, nouvelles, iris.target_names)
    }))
