"""
AI-Based Road Damage Detection & Inspection History Database Module
Persistent SQLite database engine for storing and retrieving inspection audit records.
"""

import os
import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


class InspectionDatabase:
    """
    Manages persistent SQLite storage for road damage inspection audits at data/road_damage.db.
    """
    def __init__(self, db_path: str = "data/road_damage.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self.init_database()

    def _ensure_db_dir(self):
        """
        Ensure parent directory for database exists automatically.
        """
        dir_path = os.path.dirname(self.db_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """
        Open connection to SQLite database.
        """
        self._ensure_db_dir()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """
        Automatically create the inspections table if it does not exist.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            total_detections INTEGER NOT NULL,
            potholes INTEGER NOT NULL,
            cracks INTEGER NOT NULL,
            low_severity INTEGER NOT NULL,
            medium_severity INTEGER NOT NULL,
            high_severity INTEGER NOT NULL,
            risk_score REAL NOT NULL
        );
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_query)
            conn.commit()

    def save_inspection(
        self,
        file_name: str,
        total_detections: int,
        potholes: int,
        cracks: int,
        low_severity: int,
        medium_severity: int,
        high_severity: int,
        risk_score: float,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Insert an inspection record into SQLite database.
        
        Returns:
            inserted_id: Integer primary key ID of inserted row.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        insert_query = """
        INSERT INTO inspections (
            file_name, timestamp, total_detections, potholes, cracks,
            low_severity, medium_severity, high_severity, risk_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(insert_query, (
                file_name,
                timestamp,
                int(total_detections),
                int(potholes),
                int(cracks),
                int(low_severity),
                int(medium_severity),
                int(high_severity),
                float(risk_score)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_inspections(self) -> pd.DataFrame:
        """
        Retrieve all inspection logs ordered by ID descending as a Pandas DataFrame.
        """
        query = "SELECT * FROM inspections ORDER BY id DESC;"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            return df

    def get_total_count(self) -> int:
        """
        Get total count of recorded inspections.
        """
        query = "SELECT COUNT(*) FROM inspections;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            count = cursor.fetchone()[0]
            return count

    def clear_all_inspections(self) -> bool:
        """
        Delete all inspection logs from database table.
        """
        query = "DELETE FROM inspections;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            return True
