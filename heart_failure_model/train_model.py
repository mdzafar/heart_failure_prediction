import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from heart_failure_model.data_pipeline import get_preprocessor


def load_config():
    with open("heart_failure_model/config.yml", "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    df = pd.read_csv(config['data']['path'])
    X = df.drop(config['data']['target_column'], axis=1)
    y = df[config['data']['target_column']]

    cat_cols = config['data']['categorical_columns']
    num_cols = config['data']['numerical_columns']

    preprocessor = get_preprocessor(num_cols, cat_cols)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            random_state=config['model']['random_state']
        ))
    ])

    model.fit(X, y)
    joblib.dump(model, config['model']['path'])


if __name__ == "__main__":
    train()