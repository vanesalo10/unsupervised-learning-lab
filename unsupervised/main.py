import io
import json
from PIL import Image
from fastapi import File,FastAPI
import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from unsupervised.PCA import PCA

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

filename = "../MNIST_trained_model.pickle"

# load model
loaded_model = pickle.load(open(filename, "rb"))

# Load MNIST dataset
X, y  = fetch_openml(data_id=554, parser='auto', return_X_y=True) # https://www.openml.org/d/554

# Select only 0s and 8s
num=['0','8']
y1 = y.isin(num)
indices = y1[y1].index
X = X.loc[indices]
y = y.loc[indices]

# Reset index after class selection
X.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=2)
fit_pca = pca.fit(X_train)

app = FastAPI()


@app.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
  input_image =Image.open(io.BytesIO(file)).convert("L").resize((28, 28))
  image_array = np.array(input_image)

  pca_image = fit_pca.transform(image_array.flatten())

  results = loaded_model.predict(pca_image.reshape(1,-1))
  response = '{"predicted_value":"'+str(results)+'", "model": "MNIST_trained_model"}'
  results_json = json.loads(response)
  return {"result": results_json}