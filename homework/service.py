import bentoml
import numpy as np
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


tag = "mlzoomcamp_homework:qtzdz3slg6mwwdu5"
model_ref = bentoml.sklearn.get(tag)


model_runner = model_ref.to_runner()

svc = bentoml.Service("classifier", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    return result  
