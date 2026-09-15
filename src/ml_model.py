import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

MODEL_PATH = "models/job_role_model.pkl"


def load_training_data():
    df = pd.read_csv("data/training_data.csv")

    df = df.dropna(subset=["text", "job_role"])
    df = df.drop_duplicates()

    return df


def create_models():
    models = {
        "Naive Bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                MultinomialNB(alpha=0.1)
            )
        ]),

        "Logistic Regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]),

        "Linear SVM": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    random_state=42
                )
            )
        ])
    }

    return models


def train_model():

    df = load_training_data()

    X = df["text"]
    y = df["job_role"]

    print(f"Training samples: {len(df)}")
    print(f"Job roles: {y.nunique()}")

    models = create_models()

    # ---------------------------------------------------------
    # 1. HOLD-OUT TEST SET
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = {}

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # ---------------------------------------------------------
    # 2. TRAIN AND TEST EACH MODEL
    # ---------------------------------------------------------

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        print()
        print(name)
        print("-" * 40)

        print(f"Accuracy : {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall   : {recall * 100:.2f}%")
        print(f"Macro F1 : {f1 * 100:.2f}%")

        print()
        print("Classification Report:")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

    # ---------------------------------------------------------
    # 3. 5-FOLD CROSS VALIDATION
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_results = {}

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro"
    }

    for name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        mean_accuracy = scores["test_accuracy"].mean()
        mean_precision = scores["test_precision"].mean()
        mean_recall = scores["test_recall"].mean()
        mean_f1 = scores["test_f1"].mean()

        cv_results[name] = {
            "accuracy": mean_accuracy,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": mean_f1
        }

        print()
        print(name)
        print("-" * 40)

        print(
            f"CV Accuracy : {mean_accuracy * 100:.2f}%"
        )

        print(
            f"CV Precision: {mean_precision * 100:.2f}%"
        )

        print(
            f"CV Recall   : {mean_recall * 100:.2f}%"
        )

        print(
            f"CV Macro F1 : {mean_f1 * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # 4. SELECT BEST MODEL USING MACRO F1
    # ---------------------------------------------------------

    best_model_name = max(
        cv_results,
        key=lambda name: cv_results[name]["f1"]
    )

    best_model = models[best_model_name]

    # Train best model on the complete dataset
    best_model.fit(X, y)

    # ---------------------------------------------------------
    # 5. CONFUSION MATRIX ON HOLD-OUT TEST SET
    # ---------------------------------------------------------

    best_test_predictions = results[
        best_model_name
    ]["model"].predict(X_test)

    matrix = confusion_matrix(
        y_test,
        best_test_predictions,
        labels=sorted(y.unique())
    )

    print()
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model: {best_model_name}")

    print(
        f"Test Accuracy: "
        f"{results[best_model_name]['accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Macro F1: "
        f"{results[best_model_name]['f1'] * 100:.2f}%"
    )

    print(
        f"CV Macro F1: "
        f"{cv_results[best_model_name]['f1'] * 100:.2f}%"
    )

    print()
    print("Confusion Matrix:")

    print(matrix)

    # ---------------------------------------------------------
    # 6. SAVE BEST MODEL
    # ---------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print()
    print("Model saved to:", MODEL_PATH)

    return (
        best_model,
        results,
        cv_results,
        matrix
    )


def load_model():

    return joblib.load(MODEL_PATH)


def predict_roles(model, resume_text):

    # Models with probability support
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            [resume_text]
        )[0]

        classes = model.classes_

        results = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        return results

    # Fallback for Linear SVM
    decision_scores = model.decision_function(
        [resume_text]
    )[0]

    classes = model.classes_

    results = sorted(
        zip(classes, decision_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return results