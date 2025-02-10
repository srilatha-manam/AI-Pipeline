import os
import yaml
import logging
import psycopg2
from datetime import datetime

# Load DB Config from YAML
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configurations", "config.yaml")

def load_db_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config["localdb"]["DB_CONFIG"]

# Custom PostgreSQL Handler for logging
class PostgresHandler(logging.Handler):
    def __init__(self, db_config):
        super().__init__()
        self.db_config = db_config
        self.conn = psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["dbname"],
            user=self.db_config["user"],
            password=self.db_config["password"]
        )
        self.cursor = self.conn.cursor()

    def emit(self, record):
        try:
            log_message = self.format(record)
            
            # Convert the numeric timestamp (Unix timestamp) to a proper Python datetime
            timestamp = datetime.fromtimestamp(record.created)

            query = """INSERT INTO logs (timestamp, log_level, message) VALUES (%s, %s, %s)"""
            self.cursor.execute(query, (timestamp, record.levelname, log_message))
            self.conn.commit()
        except Exception as e:
            print(f"Error writing to database: {e}")

# Configure the logger
def configure_logger():
    logger = logging.getLogger("ModelLogger")
    logger.setLevel(logging.INFO)

    # Load DB config
    db_config = load_db_config()

    # Add the custom PostgreSQL handler to the logger
    postgres_handler = PostgresHandler(db_config)
    postgres_handler.setLevel(logging.INFO)

    # Set the log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    postgres_handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(postgres_handler)

    return logger
