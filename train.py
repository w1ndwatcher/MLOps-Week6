import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import os

#  Configuration
MODEL_TRACKING_URI = "http://35.225.19.3:8100/"
MODEL_NAME = "IRIS-classifier-dt-w6"
LOCAL_MODEL_PATH = "model/model.joblib"

mlflow.set_tracking_uri(MODEL_TRACKING_URI)
mlflow.set_experiment("IRIS Classifier: Mlflow Assignment Week 6")

# load the dataset
print("Loading data...")
data = pd.read_csv('data/iris.csv')
print("Data loaded successfully!")
print(data.head())

# train test splitting
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species


# train multiple models with hyperparameter tuning and log to mlflow
param_grid = [
    {"max_depth": 2, "random_state": 1},
    {"max_depth": 3, "random_state": 1},
    {"max_depth": 4, "random_state": 1}
]

for params in param_grid:
    with mlflow.start_run():
        # Train model
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("Training Info", "Decision Tree Model for IRIS dataset")

        # Register model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_train
        )

        print(f"Run complete for params={params}, Accuracy={acc:.3f}")

print("All experiments logged to MLflow successfully!")

# fetch best model version from mlflow registry
print("Fetching best model version from MLflow Registry...")

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

best_version = None
best_accuracy = -1.0

for v in versions:
    run_id = v.run_id
    metrics = client.get_run(run_id).data.metrics
    accuracy = metrics.get("accuracy", 0)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_version = v

if not best_version:
    raise ValueError(f"No registered versions of model '{MODEL_NAME}' found.")

print(f"Best model version: {best_version.version} with accuracy={best_accuracy:.3f}")

# Load and save best model locally
best_model_uri = f"models:/{MODEL_NAME}/{best_version.version}"
best_model = mlflow.pyfunc.load_model(model_uri=best_model_uri)

os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
joblib.dump(best_model, LOCAL_MODEL_PATH)
print(f"Best model saved to {LOCAL_MODEL_PATH}")
print("Done! FastAPI will now use this best model for inference.")