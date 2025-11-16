from sklearn.ensemble import RandomForestClassifier
from helper import load_and_split
import mlflow
import mlflow.sklearn  
import os

print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

data_path = "transactions_preprocessing/metaverse_clean.csv"
print(f"Data path exists: {os.path.exists(data_path)}")

try:
    X_train, X_test, y_train, y_test = load_and_split(data_path)
    print("Data loaded successfully")
    print(f"Training data shape: {X_train.shape}")
    
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        mlflow.log_param("n_estimators", 505)
        mlflow.log_param("max_depth", 37)

        model = RandomForestClassifier(
            n_estimators=505,
            max_depth=37,
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy}")

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="metaverse_model"
        )
        print("Model logged successfully")
        
except Exception as e:
    print(f"Error in modelling: {str(e)}")
    raise