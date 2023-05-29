import os
import logging
import json
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType



def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
   
    model_path = os.getenv("AZUREML_MODEL_DIR")
    print("Model path:", model_path)
    # Load the model
    model = joblib.load(os.path.join(model_path, "model.joblib"))
    logging.info("Init complete")

# Sample input/output data for schema declaration
input_sample = pd.DataFrame({
    "cylinders": pd.Series([0], dtype="int64"),
    "displacement": pd.Series([0], dtype="float64"),
    "horsepower": pd.Series([0], dtype="int64"),
    "weight": pd.Series([0], dtype="int64"),
    "acceleration": pd.Series([0], dtype="float64"),
    "model year": pd.Series([0.0], dtype="int64")
})
output_sample = np.array([0])

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))


def run(data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()

