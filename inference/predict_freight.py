from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data: dict):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model = load_model()

    try:
        input_df = pd.DataFrame(input_data)
    except ValueError:
        input_df = pd.DataFrame([input_data])

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        missing_features = [f for f in expected_features if f not in input_df.columns]
        if missing_features:
            raise ValueError(
                f"Missing required feature(s): {missing_features}. "
                f"Expected features: {expected_features}"
            )
        input_df = input_df[expected_features]

    input_df["Predicted_Freight"] = model.predict(input_df).round(2)

    return input_df


if __name__ == "__main__":
    sample_data = {
        "Dollars": [18500, 9000]
    }

    result = predict_freight_cost(sample_data)
    print(result)