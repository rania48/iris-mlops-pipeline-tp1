import pandas as pd
from sklearn.datasets import load_iris

from drift import detecter_drift, simuler_nouvelles_donnees


def donnees_reference():
    iris = load_iris()
    return pd.DataFrame(iris.data, columns=iris.feature_names)


def test_pas_de_drift_sur_les_memes_donnees():
    reference = donnees_reference()

    rapport = detecter_drift(reference, reference)

    assert not rapport["alerte"].any()


def test_drift_detecte_sur_les_petales():
    reference = donnees_reference()

    rapport = detecter_drift(reference, simuler_nouvelles_donnees(reference))

    assert rapport.loc["petal length (cm)", "alerte"]
    assert not rapport.loc["sepal width (cm)", "alerte"]
