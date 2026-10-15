
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Local default: project folder/ecovision.db
# Render persistent disk: set DATABASE_PATH in Environment
_configured_path = os.environ.get("DATABASE_PATH")

DATABASE_PATH = (
    Path(_configured_path).expanduser()
    if _configured_path
    else BASE_DIR / "ecovision.db"
)

DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    """Create a SQLite connection with a busy timeout."""
    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=15
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_columns(connection, table_name):
    """Return column names for an existing table."""
    return {
        row["name"]
        for row in connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
    }


# --------------------------------------------------
# CREATE TABLES AND MIGRATE OLD DATABASES
# --------------------------------------------------

def init_db():
    """Create tables and add missing columns without deleting data."""

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
                action TEXT DEFAULT
                    'Completed an eco-friendly action',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add missing columns to older databases.
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
                "action": (
                    "TEXT DEFAULT "
                    "'Completed an eco-friendly action'"
                ),
                "created_at": "TIMESTAMP",
            },
        }

        for table, columns in migrations.items():
            existing = get_columns(connection, table)

            for column, column_type in columns.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE {table} "
                        f"ADD COLUMN {column} {column_type}"
                    )


# --------------------------------------------------
# TIMESTAMP HELPER
# --------------------------------------------------

def current_timestamp():
    """Return a consistent UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


# --------------------------------------------------
# SAVE NATURE SCANNER RESULTS
# --------------------------------------------------

def save_scan(
    label,
    nature_category=None,
    confidence=None,
    image_url=None
):
    """Save a successful image classification."""

    if not label or not str(label).strip():
        raise ValueError("A scan label is required.")

    if confidence is not None:
        confidence = float(confidence)

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO scans (
                label,
                nature_category,
                confidence,
                image_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(label).strip(),
            (
                str(nature_category).strip()
                if nature_category is not None
                else None
            ),
            confidence,
            str(image_url) if image_url is not None else None,
            current_timestamp(),
        ))

        return cursor.lastrowid


# --------------------------------------------------
# SAVE LAND RISK PREDICTIONS
# --------------------------------------------------

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
    """Save one land-risk prediction."""

    if not risk or not str(risk).strip():
        raise ValueError("A risk value is required.")

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO predictions (
                risk,
                vegetation,
                rainfall,
                soil_quality,
                human_activity,
                temperature,
                pollution,
                green_cover,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(risk).strip(),
            vegetation,
            rainfall,
            soil_quality,
            human_activity,
            temperature,
            pollution,
            green_cover,
            current_timestamp(),
        ))

        return cursor.lastrowid


# --------------------------------------------------
# SAVE ECO-FRIENDLY ACTIONS
# --------------------------------------------------

def save_eco_action(
    action="Completed an eco-friendly action"
):
    """Record an eco-friendly action."""

    action = str(
        action or "Completed an eco-friendly action"
    ).strip()

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO eco_actions (action, created_at)
            VALUES (?, ?)
        """, (
            action,
            current_timestamp(),
        ))

        return cursor.lastrowid


# --------------------------------------------------
# GET ENVIRONMENTAL DASHBOARD STATISTICS
# --------------------------------------------------

def get_dashboard_stats():
    """Return dashboard counters, risk counts, and recent scans."""

    with get_connection() as connection:

        # Total successful image scans.
        total_scans = connection.execute("""
            SELECT COUNT(*)
            FROM scans
        """).fetchone()[0]

        # Number of distinct non-empty nature categories.
        nature_categories = connection.execute("""
            SELECT COUNT(DISTINCT TRIM(nature_category))
            FROM scans
            WHERE nature_category IS NOT NULL
              AND TRIM(nature_category) != ''
        """).fetchone()[0]

        # Count high-risk land predictions.
        high_risk_count = connection.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE LOWER(TRIM(risk)) = 'high'
        """).fetchone()[0]

        # Total recorded eco-actions.
        eco_actions = connection.execute("""
            SELECT COUNT(*)
            FROM eco_actions
        """).fetchone()[0]

        # Risk distribution.
        risk_rows = connection.execute("""
            SELECT LOWER(TRIM(risk)) AS risk_name,
                   COUNT(*) AS total
            FROM predictions
            GROUP BY LOWER(TRIM(risk))
        """).fetchall()

        risks = {
            "Low": 0,
            "Medium": 0,
            "High": 0
        }

        for row in risk_rows:
            risk_name = str(
                row["risk_name"] or ""
            ).capitalize()

            if risk_name in risks:
                risks[risk_name] = row["total"]

        # Most recent five successful scans.
        recent_rows = connection.execute("""
            SELECT
                label,
                nature_category,
                confidence,
                image_url,
                created_at
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
        "natureCategories": nature_categories,
        "highRiskCount": high_risk_count,
        "ecoActions": eco_actions,
        "risks": risks,
        "recentScans": recent_scans,
    }


# --------------------------------------------------
# INITIALIZE DATABASE
# --------------------------------------------------

# Create tables and safely migrate existing databases.
init_db()
