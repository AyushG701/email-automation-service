import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    APP_KEY = os.getenv("SUPERTRUCK_M2M_TOKEN")
    RABBITMQ_HOST: str = os.getenv("RABBITMQ_HOST")
    RABBITMQ_PORT: str = os.getenv("RABBITMQ_PORT")
    RABBITMQ_USER: str = os.getenv("RABBITMQ_USER")
    RABBITMQ_PASS: str = os.getenv("RABBITMQ_PASS")
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
    MULTI_INTENT_ENABLED: bool = os.getenv("MULTI_INTENT_ENABLED", "True").lower() == "true"

    # Database Configuration
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", 5432)
    DB_NAME = os.getenv("DB_NAME", "email_automation")
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Model Configuration
    DEFAULT_MODEL = 'gpt-4o-mini'
    DEFAULT_TEMPERATURE = 0.7

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # SERVICES URL
    SUPERTRUCK_SERVICE_URL = os.getenv("SUPERTRUCK_SERVICE_URL")
    LOAD_SERVICE_URL = os.getenv("LOAD_SERVICE_URL")
    BROKER_SERVICE_URL = os.getenv("BROKER_SERVICE_URL")

