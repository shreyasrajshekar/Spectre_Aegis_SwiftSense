import time
import os
import logging
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.db_logger import DBLogger, build_logger

# Setup logging to see results
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def test_limit():
    # 1. Load .env
    _env_path = os.path.join(os.path.dirname(__file__), '.env')
    _env = {}
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    _env[_k.strip()] = _v.strip()

    host = _env.get("DB_HOST", "localhost")
    user = _env.get("DB_USER", "root")
    password = _env.get("DB_PASS", "") # Fix for empty password in test
    database = _env.get("DB_NAME", "aegis")
    port = int(_env.get("DB_PORT", 3306))

    print(f"Connecting to {user}@{host}:{port}/{database}...")
    
    logger = build_logger(host, user, password, database, port)
    
    if hasattr(logger, 'log') and not hasattr(logger, '_drain_loop'):
        print("ERROR: NullLogger returned. DB credentials might be wrong or DB is down.")
        return

    print("Injecting 600 telemetry rows...")
    for i in range(600):
        telemetry = {
            "latency_ms": 1.0,
            "channel_idx": 1,
            "class_name": f"Test-{i}",
            "is_busy": False,
            "reward": 0.0,
            "pls_score": 100,
            "prediction_horizon": 0.0,
            "trend": "stable",
            "radar_map": [],
            "reasoning_msg": f"Verification inject {i}"
        }
        logger.log(telemetry)
        if i % 100 == 0:
            print(f"  Injected {i}...")

    print("Waiting for background worker to drain and prune (5 seconds)...")
    time.sleep(5)
    
    # 2. Check count manually
    import mysql.connector
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, port=port)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM aegis_logs")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    print(f"Final Row Count: {count}")
    if count == 500:
        print("SUCCESS: Rule strictly enforced at 500 rows.")
    elif count < 505:
        print(f"ACCEPTABLE: {count} rows (close enough to 500).")
    else:
        print(f"FAILURE: Table contains {count} rows!")

    logger.shutdown()

if __name__ == "__main__":
    test_limit()
