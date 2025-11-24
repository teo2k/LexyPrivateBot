from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bot_token: str | None = Field(default=None, alias="BOT_TOKEN")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    vector_store_url: str | None = Field(default=None, alias="VECTOR_STORE_URL")

    debug: bool = Field(default=False, alias="DEBUG")

    # Pinecone
    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="lexy-legal-norms", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
