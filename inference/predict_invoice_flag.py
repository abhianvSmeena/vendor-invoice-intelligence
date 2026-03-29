from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "predict_flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flagging model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_invoice_flag(input_data: dict):
    """
    Predict whether an invoice should be flagged for manual approval.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()

    try:
        input_df = pd.DataFrame(input_data)
    except ValueError:
        input_df = pd.DataFrame([input_data])

    input_df["Predicted_Flag"] = model.predict(input_df).astype(int)

    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0],
    }

    result = predict_invoice_flag(sample_data)
    print(result)