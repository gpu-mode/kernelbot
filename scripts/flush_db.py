#!/usr/bin/env python3

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def flush_database():
    # Load environment variables
    load_dotenv()

    DATABASE_URL = os.getenv("DATABASE_URL")

    if DATABASE_URL is None:
        print("❌ Missing DATABASE_URL environment variable")
        return

    try:
        # Connect to database using SQLAlchemy
        print("📡 Connecting to database...")
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            with connection.begin():
                # Drop existing tables
                print("🗑️  Dropping existing tables...")
                drop_tables_query = text("""
                DROP TABLE IF EXISTS submissions CASCADE;
                DROP TABLE IF EXISTS leaderboard CASCADE;
                DROP TABLE IF EXISTS runinfo CASCADE;
                DROP TABLE IF EXISTS _yoyo_log CASCADE;
                DROP TABLE IF EXISTS _yoyo_migration CASCADE;
                DROP TABLE IF EXISTS _yoyo_version CASCADE;
                DROP TABLE IF EXISTS yoyo_lock CASCADE;
                DROP SCHEMA IF EXISTS leaderboard CASCADE;
                """)
                connection.execute(drop_tables_query)
                
        print("✅ Database flushed and recreated successfully!")

    except SQLAlchemyError as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("🔌 Database operation completed")


if __name__ == "__main__":
    flush_database()
