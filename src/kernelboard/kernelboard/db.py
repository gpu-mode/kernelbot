import psycopg2
import os
from flask import g, Flask


def get_db_connection() -> psycopg2.extensions.connection:
    """
    Get a database connection from the `g` object. If the connection is not
    already in the `g` object, create a new connection and store it in the
    `g` object.
    """
    if 'db_connection' not in g:
        g.db_connection = psycopg2.connect(os.getenv('DATABASE_URL'))
    return g.db_connection


def close_db_connection(e=None):
    """
    Close the database connection from the `g` object.
    """
    db = g.pop('db_connection', None)
    if db is not None:
        db.close()


def init_app(app: Flask):
    app.teardown_appcontext(close_db_connection)