# src/models/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import joblib
import os


from src.config import (
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    MLFLOW_EXPERIMENT,
    MODEL_ARTIFACT_PATH,
    INFERENCE_THRESHOLD,
)
from src.data.pipeline import create_preprocessing_pipeline
from src.evaluation.metrics import calculate_metrics


def prepare_data(df: pd.DataFrame):

    # Aceita os dois nomes possíveis do target
    col_target = TARGET if TARGET in df.columns else "Churn Value"

    y = df[col_target].replace({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[col_target])
    return X, y


def split_data(X, y):
    # Divide dados em treino, validação e teste com stratify para manter proporção do target.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"  Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_full_pipeline(model) -> Pipeline:
    # from src.data.pipeline import create_preprocessing_pipeline as create_prep

    # Junta o pipeline de pré-processamento com o model.
    preprocessamento = create_preprocessing_pipeline()

    pipeline_completo = Pipeline(
        steps=[
            *preprocessamento.steps,  # desempacota todas as etapas de pré-proc
            ("model", model),  # adiciona o model no final
        ]
    )

    return pipeline_completo


def train_pipeline(
    model,
    nome_run: str,
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_meta: dict,
    fazer_cv: bool = True,
):
    # Treina um model usando o pipeline completo e loga no MLflow.
    pipeline = create_preprocessing_pipeline(model)

    mlflow.end_run()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=nome_run) as run:
        mlflow.log_params(dataset_meta)

        if fazer_cv:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=kf, scoring="roc_auc"
            )
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Treino
        pipeline.fit(X_train, y_train)

        #  Avaliação
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "pipeline")  # salva o pipeline inteiro

        print(f"\n=== {nome_run.upper()} ===")
        for nome, valor in metrics.items():
            print(f"  {nome:<16}: {valor:.4f}")

        artifact = {
            "pipeline": pipeline,
            "threshold": INFERENCE_THRESHOLD,
            "metadata": {**dataset_meta, "run_name": nome_run, "run_id": run.info.run_id, },
        }

    path_run = MODEL_ARTIFACT_PATH.replace(".joblib", f"_{nome_run}.joblib")
    os.makedirs(os.path.dirname(path_run), exist_ok=True)
    joblib.dump(artifact, path_run)

    # Salva também como "latest" para a API consumir
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)
    print(f"  Artifact salvo em: {path_run}")
    print(f"  Artifact latest:   {MODEL_ARTIFACT_PATH}")

    return pipeline, y_pred, y_proba





def get_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    # Aplica o pipeline de pré-processamento (sem model) para obter os dados escalados.

    preprocessor = create_preprocessing_pipeline()  # ok
    preprocessor.fit(X_train, y_train)

    X_train_sc = preprocessor.transform(X_train)
    X_val_sc = preprocessor.transform(X_val)
    X_test_sc = preprocessor.transform(X_test)

    return X_train_sc, X_val_sc, X_test_sc


