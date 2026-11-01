import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris

# Charge le modèle depuis le Model Registry grâce à son alias
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model("models:/Iris_Model@champion")

iris = load_iris()

sample = [iris.data[0]]

prediction = model.predict(sample)

print("Modèle : Iris_Model@champion")
print("Classe :", prediction[0])
print("Nom :", iris.target_names[prediction[0]])
