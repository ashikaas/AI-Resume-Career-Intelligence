from src.ml_model import train_model


model, results, cv_results, matrix = train_model()


print()
print("=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print()
print("Hold-out Test Performance:")

for name, result in results.items():

    print(
        f"{name}: "
        f"Accuracy={result['accuracy'] * 100:.2f}% | "
        f"Macro F1={result['f1'] * 100:.2f}%"
    )


print()
print("5-Fold Cross-Validation:")

for name, result in cv_results.items():

    print(
        f"{name}: "
        f"Accuracy={result['accuracy'] * 100:.2f}% | "
        f"Macro F1={result['f1'] * 100:.2f}%"
    )


best_model_name = max(
    cv_results,
    key=lambda name: cv_results[name]["f1"]
)

print()
print("=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print(f"Best Model: {best_model_name}")

print(
    f"Cross-Validation Macro F1: "
    f"{cv_results[best_model_name]['f1'] * 100:.2f}%"
)

print()
print("Model saved successfully!")
print("Location: models/job_role_model.pkl")