import logging
import os
import sqlite3

from injector import Module, provider

from .constants import Configuration, user_dir


class DbModule(Module):
    def __init__(self):
        self._db_initialized = False

    def _initialize_db(self, db: sqlite3.Connection):
        cursor = db.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS tweeter ('
                       '  user_id INTEGER PRIMARY KEY, '
                       '  screen_name TEXT, '
                       '  oldest_tweet_id INTEGER'
                       ')')
        cursor.execute('CREATE TABLE IF NOT EXISTS tweet ('
                       '  user_id INTEGER, '
                       '  tweet_text TEXT,'
                       '  FOREIGN KEY(user_id) REFERENCES tweeter(user_id)'
                       ')')
        db.commit()

    def _get_database(self, config):
        result = config.get('DB connection')
        if result is None:
            result = os.path.join(user_dir, 'my-bot.db')
        else:
            result = os.path.expanduser(result)
        return result

    @provider
    def provide_db_connection(self, config: Configuration, logger: logging.Logger) -> sqlite3.Connection:
        database = self._get_database(config)
        logger.debug("Database: %s", database)
        result = sqlite3.connect(database)
        if not self._db_initialized:
            self._initialize_db(result)
            self._db_initialized = True
        return result
