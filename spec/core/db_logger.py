"""
db_logger.py – Async MySQL telemetry sink for Spectre AEGIS.

Design
------
* A background daemon thread drains a queue.Queue, so the 20 Hz RT loop
  is never blocked by a slow DB write.
* The table is created automatically on first connection.
* The connection is re-established transparently if it drops.
* If no credentials are provided the logger is a no-op (safe default).
"""

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS aegis_logs (
    id                 BIGINT       AUTO_INCREMENT PRIMARY KEY,
    ts                 DATETIME(3)  NOT NULL,
    latency_ms         FLOAT,
    channel_idx        TINYINT,
    class_name         VARCHAR(64),
    confidence         FLOAT,
    is_busy            TINYINT(1),
    reward             FLOAT,
    pls_score          SMALLINT,
    event_trigger      VARCHAR(32),
    prediction_horizon FLOAT,
    trend              VARCHAR(16),
    radar_targets      TINYINT,
    eco_saving         TINYINT,
    last_hop_db        VARCHAR(16),
    current_layer      VARCHAR(16),
    reasoning_msg      VARCHAR(255)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

"""

_INSERT_SQL = """
INSERT INTO aegis_logs (
    ts, latency_ms, channel_idx, class_name, confidence,
    is_busy, reward, pls_score, event_trigger,
    prediction_horizon, trend, radar_targets,
    eco_saving, last_hop_db, current_layer, reasoning_msg
) VALUES (
    %(ts)s, %(latency_ms)s, %(channel_idx)s, %(class_name)s, %(confidence)s,
    %(is_busy)s, %(reward)s, %(pls_score)s, %(event_trigger)s,
    %(prediction_horizon)s, %(trend)s, %(radar_targets)s,
    %(eco_saving)s, %(last_hop_db)s, %(current_layer)s, %(reasoning_msg)s
)
"""

# Pruning is now handled with formatted integers (as LIMIT doesn't support %s parameters in many MySQL versions)
_PRUNE_OLD_SQL = "DELETE FROM aegis_logs ORDER BY id ASC LIMIT %d"
_COUNT_SQL     = "SELECT COUNT(*) FROM aegis_logs"


class DBLogger:
    """
    Non-blocking MySQL telemetry logger.

    Parameters
    ----------
    host     : MySQL host  (e.g. 'localhost')
    port     : MySQL port  (default 3306)
    user     : MySQL user
    password : MySQL password
    database : Target database / schema name
    queue_maxsize : Max pending rows before the oldest is silently dropped
    """

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        queue_maxsize: int = 2000,
    ):
        self._config = dict(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connection_timeout=5,
            autocommit=True,
        )
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._conn = None
        self._enabled = True

        # Bootstrap connection & schema, then hand off to background thread
        self._connect()

        self._worker = threading.Thread(
            target=self._drain_loop, name="aegis-db-logger", daemon=True
        )
        self._worker.start()
        log.info(
            f"DBLogger: connected to {user}@{host}:{port}/{database} "
            f"– background writer active."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, telemetry: dict) -> None:
        """Enqueue a telemetry snapshot for async DB insertion."""
        if not self._enabled:
            return

        radar_map = telemetry.get("radar_map", [])
        radar_targets = len(radar_map) if isinstance(radar_map, list) else 0

        row = {
            "ts":                 datetime.now(),
            "latency_ms":         telemetry.get("latency_ms"),
            "channel_idx":        telemetry.get("channel_idx"),
            "class_name":         str(telemetry.get("class_name", ""))[:64],
            "confidence":         telemetry.get("confidence"),
            "is_busy":            int(bool(telemetry.get("is_busy", False))),
            "reward":             telemetry.get("reward"),
            "pls_score":          telemetry.get("pls_score"),
            "event_trigger":      str(telemetry.get("event_trigger") or "")[:32] or None,
            "prediction_horizon": telemetry.get("prediction_horizon"),
            "trend":              str(telemetry.get("trend", ""))[:16],
            "radar_targets":      radar_targets,
            "eco_saving":         telemetry.get("eco_saving"),
            "last_hop_db":        str(telemetry.get("last_hop_db", "0.00"))[:16],
            "current_layer":      str(telemetry.get("current_layer", ""))[:16],
            "reasoning_msg":      str(telemetry.get("reasoning_msg", ""))[:255],
        }

        try:
            self._q.put_nowait(row)
        except queue.Full:
            # Drop oldest to prevent unbounded growth under a slow DB
            try:
                self._q.get_nowait()
                self._q.put_nowait(row)
            except queue.Empty:
                pass

    def shutdown(self) -> None:
        """Flush remaining rows and close the connection gracefully."""
        if not self._enabled:
            return
        log.info("DBLogger: flushing queue before shutdown…")
        self._enabled = False
        # Push sentinel to stop worker
        self._q.put(None)
        # Wait for worker thread to exit
        self._worker.join(timeout=2.0)
        
        if self._conn and self._conn.is_connected():
            self._conn.close()
        log.info("DBLogger: closed.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Try to open (or re-open) the MySQL connection. Returns success."""
        try:
            import mysql.connector  # lazy import – optional dependency
            if self._conn and self._conn.is_connected():
                self._conn.close()
            self._conn = mysql.connector.connect(**self._config)
            cursor = self._conn.cursor()
            cursor.execute(_CREATE_TABLE_SQL)
            cursor.close()
            return True
        except Exception as exc:
            log.error(f"DBLogger: connection failed – {exc}")
            self._conn = None
            return False

    def _drain_loop(self) -> None:
        """Background thread: dequeue rows and INSERT them."""
        RETRY_DELAY = 5  # seconds between reconnect attempts
        MAX_RETRIES = 5  # Don't stay stuck forever if DB is broken

        while True:
            row = self._q.get()  # blocks until a row is available
            if row is None:
                self._q.task_done()
                break # Shutdown signal
            
            inserted = False
            retry_count = 0
            while not inserted:
                if self._conn is None or not self._conn.is_connected():
                    log.warning(f"DBLogger: reconnecting (Attempt {retry_count+1}/{MAX_RETRIES})…")
                    if not self._connect():
                        retry_count += 1
                        if retry_count >= MAX_RETRIES:
                            log.error("DBLogger: Maximum reconnect attempts reached. Dropping row.")
                            break
                        time.sleep(RETRY_DELAY)
                        continue  # keep retrying
                
                try:
                    cursor = self._conn.cursor()
                    cursor.execute(_INSERT_SQL, row)
                    
                    # Enforce 500-row limit (Check count and prune oldest if exceeding)
                    try:
                        cursor.execute(_COUNT_SQL)
                        count = cursor.fetchone()[0]
                        if count > 500:
                            excess = int(count - 500)
                            cursor.execute(_PRUNE_OLD_SQL % excess)
                            log.info(f"DBLogger: Pruned {excess} oldest rows (Total: {count} -> 500)")
                    except Exception as clean_exc:
                        log.error(f"DBLogger: Pruning query failed: {clean_exc}")
                    
                    self._conn.commit()
                    cursor.close()
                    inserted = True
                except Exception as exc:
                    log.warning(f"DBLogger: Transaction failed ({exc}). Check credentials and table state.")
                    self._conn = None  # force reconnect on next iteration

            self._q.task_done()


class NullLogger:
    """Drop-in no-op replacement when no DB credentials are configured."""

    def log(self, _telemetry: dict) -> None:
        pass

    def shutdown(self) -> None:
        pass


def build_logger(
    host: Optional[str],
    user: Optional[str],
    password: Optional[str],
    database: Optional[str],
    port: int = 3306,
) -> "DBLogger | NullLogger":
    """
    Factory used by app.py.
    Returns a real DBLogger if all credentials are given, otherwise NullLogger.
    """
    if all(x is not None for x in [host, user, database]):
        try:
            return DBLogger(host=host, user=user, password=password,
                            database=database, port=port)
        except Exception as exc:
            log.error(f"DBLogger: startup failed, logging disabled – {exc}")
            return NullLogger()
    else:
        log.info("DBLogger: no DB credentials supplied – logging disabled.")
        return NullLogger()
