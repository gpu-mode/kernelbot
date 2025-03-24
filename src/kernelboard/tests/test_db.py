import pytest
import psycopg2
from kernelboard.db import get_db_connection


def test_get_db_connection(app):
    with app.app_context():
        conn = get_db_connection()
        assert conn is not None
        assert conn.closed == False
        assert conn is get_db_connection()
    
    with pytest.raises(psycopg2.InterfaceError) as e:
        conn.cursor().execute('SELECT 1')

    assert 'connection already closed' in str(e.value)