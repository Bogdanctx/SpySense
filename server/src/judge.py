import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Judge:
    def __init__(self, feature_list):
        self.pipeline = Pipeline([])  # Initialize an empty pipeline
        self.name_of_features = feature_list

    def predict_proba(self, features):
        """Return probability of the positive class."""
        return self.pipeline.predict_proba(features)

    def load_model(self, model_path):
        """Load a saved model from disk."""
        self.pipeline = joblib.load(model_path)

    def load_data(self):
        """Load train, validation, and test datasets."""
        train_csv = pd.read_csv("./data/train.csv")
        validation_csv = pd.read_csv("./data/validation.csv")
        test_csv = pd.read_csv("./data/test.csv")
        features = pd.read_csv("dataset_features.csv")

        # Merge keeps hash+label alignment
        train = pd.merge(train_csv, features, on=["hash", "label"])
        validation = pd.merge(validation_csv, features, on=["hash", "label"])
        test = pd.merge(test_csv, features, on=["hash", "label"])

        # Split into X/y
        X_train = train.drop(columns=["hash", "label"])
        y_train = train["label"].to_numpy()

        X_val = validation.drop(columns=["hash", "label"])
        y_val = validation["label"].to_numpy()

        X_test = test.drop(columns=["hash", "label"])
        y_test = test["label"].to_numpy()

        X_train = X_train[self.name_of_features].to_numpy()
        X_val = X_val[self.name_of_features].to_numpy()
        X_test = X_test[self.name_of_features].to_numpy()

        return X_train, y_train, X_val, y_val, X_test, y_test

    def print_evaluation(self, name, X, y):
        preds = self.pipeline.predict(X)
        print(f"{name} accuracy: {accuracy_score(y, preds) * 100:.2f}%")
        print(f"{name} confusion matrix:\n{confusion_matrix(y, preds)}")
        print(f"{name} classification report:\n{classification_report(y, preds, digits=4)}\n")

    
    def fit(self):
        """Train the model."""
        X_train, y_train, _, _, _, _ = self.load_data()
        self.pipeline.fit(X_train, y_train)
        self.print_evaluation("Train", X_train, y_train)

    def evaluate(self):
        """Evaluate the model's performance."""
        _, _, X_validation, y_validation, X_test, y_test = self.load_data()

        self.print_evaluation("Validation", X_validation, y_validation)
        self.print_evaluation("Test", X_test, y_test)

    def save_model(self, model_path):
        """Save the trained model to disk."""
        joblib.dump(self.pipeline, model_path)

    def load_model(self, model_path):
        """Load a saved model from disk."""
        self.pipeline = joblib.load(model_path)