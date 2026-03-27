import pickle
import numpy as np

# Class labels
labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

print("Prediction:", labels[prediction[0]])

with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nscikit-learn")
