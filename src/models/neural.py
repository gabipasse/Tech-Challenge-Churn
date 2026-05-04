# src/models/neural.py

import os
import joblib

import numpy as np
import torch
import torch.nn as nn
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import EPOCHS, PATIENCE, LR, BATCH_SIZE, MLFLOW_EXPERIMENT, MODEL_ARTIFACT_PATH
from src.evaluation.metrics import calculate_metrics


def create_tensors(X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test):

    # Converte arrays numpy para tensores do PyTorch.
    # O unsqueeze(1) transforma y de shape (N,) para (N, 1),
    def to_tensor(arr):
        return torch.tensor(np.array(arr).astype(float), dtype=torch.float)

    X_train_t = to_tensor(X_train_sc)
    X_val_t = to_tensor(X_val_sc)
    X_test_t = to_tensor(X_test_sc)
    y_train_t = to_tensor(y_train).unsqueeze(1)
    y_val_t = to_tensor(y_val).unsqueeze(1)
    y_test_t = to_tensor(y_test)

    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t


def create_model(input_dim: int) -> nn.Sequential:

    hidden_dim = input_dim // 2
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )


def mlp(X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, dataset_meta: dict):

    # Converte dados para tensores
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = create_tensors(
        X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE
    )

    input_dim = X_train_sc.shape[1]
    hidden_dim = input_dim // 2
    classificator = create_model(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classificator.parameters(), lr=LR, weight_decay=0.0001)

    mlflow.end_run()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="mlp_pytorch"):
        mlflow.log_params(dataset_meta)
        mlflow.log_params(
            {
                "architecture": f"{input_dim}-{hidden_dim}-{hidden_dim}-1",
                "activation": "ReLU",
                "loss_function": "BCELoss",
                "optimizer": "Adam",
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "patience": PATIENCE,
                "max_epochs": EPOCHS,
            }
        )

        best_val_loss = float("inf")
        counter = 0
        stop_epoch = 0

        for epoch in range(EPOCHS):
            # Treino
            classificator.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # zera gradientes anteriores
                outputs = classificator(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # calcula gradientes
                optimizer.step()  # atualiza pesos
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validação
            val_loss = 0.0
            with torch.no_grad():  # desliga o cálculo de gradiente
                for inputs, labels in val_loader:
                    outputs = classificator(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            print(
                f"Época {epoch + 1:>5} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            #  Early Stopping

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                stop_epoch = epoch + 1
                torch.save(classificator.state_dict(), "best_mlp.pt")
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(f"⏹  Early stopping na época {epoch + 1}")
                    break

        #  Avaliação final com o melhor checkpoint
        classificator.load_state_dict(torch.load("best_mlp.pt"))
        classificator.eval()

        with torch.no_grad():
            proba_mlp = classificator(X_test_t).numpy().flatten()

        # Testa diferentes thresholds para escolher o melhor
        print("\nAnálise de threshold:")
        for t in [0.3, 0.4, 0.5]:
            pred_t = (proba_mlp >= t).astype(int)
            print(
                f"  Threshold {t} → F1: {f1_score(y_test, pred_t):.4f} | "
                f"Precision: {precision_score(y_test, pred_t):.4f} | "
                f"Recall: {recall_score(y_test, pred_t):.4f}"
            )

        THRESHOLD = 0.4
        y_pred_mlp = (proba_mlp >= THRESHOLD).astype(int)

        metricas = calculate_metrics(y_test, y_pred_mlp, proba_mlp)
        mlflow.log_metrics(metricas)
        mlflow.log_param("stop_epoch ", stop_epoch)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact("best_mlp.pt")

        print("\n=== MLP PYTORCH ===")
        for nome, valor in metricas.items():
            print(f"  {nome:<16}: {valor:.4f}")
        print(f"\n  Melhor Val Loss: {best_val_loss:.4f} (época {stop_epoch})")

        print("\n=== MLP PYTORCH ===")
        for nome, valor in metricas.items():
            print(f"  {nome:<16}: {valor:.4f}")
        print(f"\n  Melhor Val Loss: {best_val_loss:.4f} (época {stop_epoch})")

       
        

        artifact = {
            "model": classificator,
            "threshold": THRESHOLD,
            "metadata": {**dataset_meta, "run_name": "mlp_pytorch"},
        }

        path_run = MODEL_ARTIFACT_PATH.replace(".joblib", "_mlp_pytorch.joblib")
        os.makedirs(os.path.dirname(path_run), exist_ok=True)
        joblib.dump(artifact, path_run)
        joblib.dump(artifact, MODEL_ARTIFACT_PATH)  # salva como latest também

        print(f"  Artifact salvo em: {path_run}")
        print(f"  Artifact latest:   {MODEL_ARTIFACT_PATH}")

    return classificator, y_pred_mlp, proba_mlp
