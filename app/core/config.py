from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "projetoPDI"
    app_env: str = "local"
    database_url: str = "teste"

    class Config:
        env_file = ".env"


settings = Settings()
