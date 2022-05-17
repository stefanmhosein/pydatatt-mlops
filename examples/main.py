import numpy as np
import pickle
import logging
import uvicorn

from typing import Union, List
from fastapi import FastAPI, Query

app = FastAPI()
logger = logging.getLogger(__name__)


@app.get("/")
def get_root():
    return {"Hello": "World"}


@app.get("/wine_quality/{input_data}")
def read_input(
    fixed_acidity: float = 12.8,
    volatile_acidity: float = 0.029,
    citric_acid: float = 0.48,
    residual_sugar: float = 0.98,
    chlorides: float = 6.2,
    free_sulfur_dioxide: float = 29,
    total_sulfur_dioxide: float = 3.33,
    density: float = 1.2,
    pH: float = 0.39,
    sulphates: float = 75,
    alcohol: float = 0.66,
):
    logger.info("logging from the root logger")
    with open(
        "file:///Users/stefanhosein/other/pydatatt-mlops/examples/sklearn_elasticnet_wine/mlruns/0/5b606f5c1186486982a60b3d5929eae3/artifacts/model/model.pkl",
        "rb",
    ) as pickle_file:
        model = pickle.load(pickle_file)
    data = [
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
    ]
    quality = model.predict(np.array(data).reshape(1, -1))[0]
    return {"predicted quality": quality}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
