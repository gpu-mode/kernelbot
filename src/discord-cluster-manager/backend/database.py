import os
from typing import Generator

import dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

dotenv.load_dotenv()

# Database configuration
DATABASE_URL = os.environ["DATABASE_URL"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_DATABASE = os.environ["POSTGRES_DATABASE"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]
DISABLE_SSL = os.getenv("DISABLE_SSL", "false").lower() == "true"


# Construct database URL if not provided
if not DATABASE_URL:
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"

# Add SSL mode
if DISABLE_SSL:
    DATABASE_URL += "?sslmode=disable"
else:
    DATABASE_URL += "?sslmode=require"

# Create engine
engine = create_engine(DATABASE_URL, echo=os.getenv("DEBUG", "false").lower() == "true")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
