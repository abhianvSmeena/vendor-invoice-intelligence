from model_evaluation import train_random_forest, evaluate_classifier
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
import joblib

FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_item_quantity',
    'total_item_dollars'
]

TARGET = "flag_invoice"


def main():

    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, 'models/scaler.pkl'
    )

    # Train model (GridSearch)
    grid_search = train_random_forest(X_train, y_train)

    # Evaluate best model
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    joblib.dump(grid_search.best_estimator_, 'models/predict_flag_invoice.pkl')


if __name__ == "__main__":
    main()