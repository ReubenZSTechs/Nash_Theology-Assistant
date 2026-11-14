import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql

load_dotenv()


class PGConnect:
    def __init__(self, dbname: str):
        self._params = {
            "host": os.getenv("PG_HOST", "localhost"),
            "port": int(os.getenv("PG_PORT", "5432")),
            "user": os.getenv("PG_USER", "nash"),
            "password": os.environ["PG_PASSWORD"],
        }
        self._dbname = dbname
        self._admin_connection = None
        self._db_connection = None


    def __enter__(self):
        return self


    def __exit__(self, *exc_info):
        self.close()


    def get_admin_connection(self) -> psycopg.Connection:
        if self._admin_connection is None or self._admin_connection.closed:
            self._admin_connection = psycopg.connect(**self._params, dbname="postgres", autocommit=True)
        return self._admin_connection


    def get_db_connection(self) -> psycopg.Connection:
        if self._db_connection is None or self._db_connection.closed:
            self.ensure_database()
            self._db_connection = psycopg.connect(**self._params, dbname=self._dbname)
            self._db_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._db_connection.commit()
            register_vector(self._db_connection)
        return self._db_connection


    def ensure_database(self) -> None:
        admin_connection = self.get_admin_connection()
        exists = admin_connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (self._dbname,),
        ).fetchone()
        if not exists:
            admin_connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self._dbname)))


    def close(self) -> None:
        for connection in (self._db_connection, self._admin_connection):
            if connection is not None and not connection.closed:
                connection.close()