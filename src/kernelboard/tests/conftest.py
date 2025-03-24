import os
import pytest
import psycopg2
from yoyo import read_migrations, get_backend
from kernelboard import create_app

def get_test_db_url():
    """Get test database URL - you might want to use environment variables"""
    return 'postgresql://postgres:postgres@localhost/kernelboard_test'


@pytest.fixture
def app():
    init_test_db()

    app = create_app({
        'TESTING': True,
        'DATABASE': os.getenv('DATABASE_URL'),
    })

    yield app


def init_test_db():
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    conn.autocommit = True
    cursor = conn.cursor()

    cursor.execute('DROP DATABASE IF EXISTS kernelboard_test')
    cursor.execute('CREATE DATABASE kernelboard_test')

    cursor.close()
    conn.close()

    # TODO was getting this working on Monday:     

    # backend = get_backend(os.getenv('DATABASE_URL'))
    # migrations = read_migrations('src/discord-cluster-manager/migrations')
    
    # with backend.lock():
    #     backend.apply_migrations(migrations)
