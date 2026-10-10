import joblib
from sklearn.datasets import load_iris

model = joblib.load("models/iris_model_100.pkl")

iris = load_iris()

sample = [iris.data[0]]

prediction = model.predict(sample)

print("Classe :", prediction[0])
print("Nom :", iris.target_names[prediction[0]])