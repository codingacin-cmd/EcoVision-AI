
import sqlite3
from pathlib import Path
from datetime import datetime
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "ecovision.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create tables and add columns missing from older databases."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                nature_category TEXT,
                confidence REAL,
                image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                risk TEXT NOT NULL,
                vegetation REAL,
                rainfall REAL,
                soil_quality REAL,
                human_activity REAL,
                temperature REAL,
                pollution REAL,
                green_cover REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS eco_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT DEFAULT 'Completed an eco-friendly action',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate existing databases without deleting saved data.
        migrations = {
            "scans": {
                "label": "TEXT",
                "nature_category": "TEXT",
                "confidence": "REAL",
                "image_url": "TEXT",
                "created_at": "TIMESTAMP",
            },
            "predictions": {
                "risk": "TEXT",
                "vegetation": "REAL",
                "rainfall": "REAL",
                "soil_quality": "REAL",
                "human_activity": "REAL",
                "temperature": "REAL",
                "pollution": "REAL",
                "green_cover": "REAL",
                "created_at": "TIMESTAMP",
            },
            "eco_actions": {
                "action": "TEXT DEFAULT 'Completed an eco-friendly action'",
                "created_at": "TIMESTAMP",
            },
        }

        for table, columns in migrations.items():
            existing_columns = {
                row["name"]
                for row in connection.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }

            for column, column_type in columns.items():
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE {table} "
                        f"ADD COLUMN {column} {column_type}"
                    )



def save_scan(label, nature_category, confidence, image_url):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO scans (
                label, nature_category, confidence, image_url, created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            label,
            nature_category,
            confidence,
            image_url,
            datetime.now().isoformat(timespec="seconds"),
        ))



def save_prediction(
    risk,
    vegetation=None,
    rainfall=None,
    soil_quality=None,
    human_activity=None,
    temperature=None,
    pollution=None,
    green_cover=None,
):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO predictions (
                risk, vegetation, rainfall, soil_quality, human_activity,
                temperature, pollution, green_cover, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            risk,
            vegetation,
            rainfall,
            soil_quality,
            human_activity,
            temperature,
            pollution,
            green_cover,
            datetime.now().isoformat(timespec="seconds"),
        ))


def save_eco_action(action="Completed an eco-friendly action"):
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO eco_actions (action) VALUES (?)",
            (action,)
        )


def get_dashboard_stats():
    with get_connection() as connection:
        total_scans = connection.execute(
            "SELECT COUNT(*) FROM scans"
        ).fetchone()[0]

        category_count = connection.execute("""
            SELECT COUNT(DISTINCT nature_category)
            FROM scans
            WHERE nature_category IS NOT NULL
              AND TRIM(nature_category) != ''
        """).fetchone()[0]

        high_risk_count = connection.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE LOWER(risk) = 'high'
        """).fetchone()[0]

        eco_actions = connection.execute(
            "SELECT COUNT(*) FROM eco_actions"
        ).fetchone()[0]

        risk_rows = connection.execute("""
            SELECT risk, COUNT(*) AS total
            FROM predictions
            GROUP BY risk
        """).fetchall()

        risks = {"Low": 0, "Medium": 0, "High": 0}

        for row in risk_rows:
            risk_name = str(row["risk"]).capitalize()
            if risk_name in risks:
                risks[risk_name] = row["total"]

        recent_rows = connection.execute("""
            SELECT label, nature_category, confidence, image_url, created_at
            FROM scans
            ORDER BY id DESC
            LIMIT 5
        """).fetchall()

        recent_scans = [
            {
                "label": row["label"],
                "nature_category": row["nature_category"],
                "confidence": row["confidence"],
                "image_url": row["image_url"],
                "created_at": row["created_at"],
            }
            for row in recent_rows
        ]

    return {
        "totalScans": total_scans,
        "natureCategories": category_count,
        "highRiskCount": high_risk_count,
        "ecoActions": eco_actions,
        "risks": risks,
        "recentScans": recent_scans,
    }


# Create or migrate the database when the module is imported.
init_db()
