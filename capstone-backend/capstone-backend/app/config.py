from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongo_uri: str = ""
    mongo_db: str = "capstone_groundedness"

    # G(c) = w1*F + w2*P + w3*D  -- placeholders until calibrated (Mohit)
    w_fact: float = 0.4
    w_policy: float = 0.3
    w_ground: float = 0.3

    pass_threshold: float = 0.7
    flag_threshold: float = 0.4
    max_regenerations: int = 2


settings = Settings()
