from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv

from app.core.config import settings


@contextmanager
def get_connection():
    load_dotenv()
    connection = psycopg2.connect(settings.database_url)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

