from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    app_name: str = "EcoVision"
    app_env: str = "local"
    yolo_model_path: str = "best.pt"
    yolo_confidence: float = 0.25
    # Tamanho de entrada da inferência. 0 = usar o tamanho salvo nos pesos
    # (o mesmo do treino), que é o recomendado.
    yolo_imgsz: int = 0
    max_image_size_bytes: int = 10 * 1024 * 1024
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    # Pasta copiada do Colab (MyDrive/EcoVision/resultados)
    results_dir: str = "resultados"

    @property
    def results_dir_path(self) -> Path:
        path = Path(self.results_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

settings = Settings()
