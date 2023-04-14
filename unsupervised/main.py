import io
import json
from PIL import Image
from fastapi import File,FastAPI
import pickle
import numpy as np


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

filename = "../MNIST_trained_model.pickle"

# load model
loaded_model = pickle.load(open(filename, "rb"))

app = FastAPI()


@app.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
  input_image =Image.open(io.BytesIO(file)).convert("L").resize((28, 28))
  image_array = np.array(input_image)
  results = loaded_model.predict(image_array.flatten().reshape(-1, len(image_array.flatten())))
  response = '{"predicted_value":"'+str(results)+'", "model": "MNIST_trained_model"}'
  results_json = json.loads(response)
  return {"result": results_json}