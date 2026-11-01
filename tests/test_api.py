import pytest

from api import app


@pytest.fixture
def client():
    return app.test_client()


def test_home(client):
    reponse = client.get("/")

    assert reponse.status_code == 200
    assert reponse.get_json()["status"] == "running"


def test_health(client):
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.get_json() == {"status": "ok", "model_loaded": True}


def test_version(client):
    reponse = client.get("/version")

    assert reponse.status_code == 200
    assert reponse.get_json()["name"] == "Iris_Model"


def test_predict_setosa(client):
    reponse = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})

    assert reponse.status_code == 200
    assert reponse.get_json()["prediction"] == 0
    assert reponse.get_json()["class_name"] == "setosa"


@pytest.mark.parametrize("corps", [
    {},
    {"features": [5.1, 3.5, 1.4]},
    {"features": [5.1, 3.5, 1.4, "a"]},
    {"features": [5.1, 3.5, 1.4, True]},
    {"features": "5.1, 3.5, 1.4, 0.2"},
    [5.1, 3.5, 1.4, 0.2],
])
def test_predict_donnees_invalides(client, corps):
    reponse = client.post("/predict", json=corps)

    assert reponse.status_code == 400
