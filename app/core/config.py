from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "EcoVision"
    app_env: str = "local"
    database_url: str = "teste"
    yolo_model_path: str = "yolo11n.pt"
    yolo_confidence: float = 0.25
    max_image_size_bytes: int = 10 * 1024 * 1024
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

settings = Settings()
