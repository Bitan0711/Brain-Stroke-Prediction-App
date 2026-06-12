import pandas as pd
import joblib

model_data = joblib.load("model.joblib")

def predict(data):

    df = pd.DataFrame([data])

    encoded_cols = model_data["encoded_cols"]
    numeric_cols = model_data["numeric_cols"]

    preprocessor = model_data["preprocessor"]

    df[encoded_cols] = preprocessor.transform(df)

    X = df[numeric_cols + encoded_cols]

    return model_data["model"].predict(X)[0]
