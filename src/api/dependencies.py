import joblib
from fastapi import HTTPException
from src.config import MODEL_ARTIFACT_PATH
from src.api.logging import build_logger

logger = build_logger("churn_api.dependencies")

# Dicionário compartilhado — populado no lifespan de app.py
MODEL_ARTIFACTS: dict = {}


def load_artifacts() -> None:
    try:
        artifacts = joblib.load(MODEL_ARTIFACT_PATH)
        MODEL_ARTIFACTS["pipeline"] = artifacts["model"]
        MODEL_ARTIFACTS["threshold"] = artifacts.get("threshold", 0.5)
        MODEL_ARTIFACTS["metadata"] = artifacts.get("metadata", {})
        logger.info("Modelo carregado", extra={"path": MODEL_ARTIFACT_PATH})
    except FileNotFoundError:
        logger.warning(
            "Arquivo de modelo não encontrado; API sobe sem modelo",
            extra={"path": MODEL_ARTIFACT_PATH},
        )


def clear_artifacts() -> None:

    MODEL_ARTIFACTS.clear()
    logger.info("Artefatos liberados")


def get_pipeline():
    pipeline = MODEL_ARTIFACTS.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique os logs de inicialização.",
        )
    return pipeline


def get_threshold() -> float:
    return MODEL_ARTIFACTS.get("threshold", 0.5)


def get_metadata() -> dict:
    return MODEL_ARTIFACTS.get("metadata", {})
