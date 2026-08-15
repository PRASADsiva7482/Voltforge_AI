"""
VoltForge AI Enterprise Database Persistence Layer.
Features auto-reconnection, retry backoff, threadpool async queueing,
safe multi-type JSON serialization, and comprehensive error shielding.
"""

import concurrent.futures
import datetime
import decimal
import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pymysql
import pymysql.cursors
import numpy as np

import config

logger = logging.getLogger("voltforge-ai.database")


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling datetimes, Decimals, NumPy arrays, sets, and UUIDs."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return super().default(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Serializes any object to JSON without throwing serialization errors."""
    try:
        return json.dumps(obj, cls=SafeJSONEncoder, ensure_ascii=False, **kwargs)
    except Exception as exc:
        logger.warning(f"safe_json_dumps fallback triggered: {exc}")
        return json.dumps(str(obj))


class DatabaseManager:
    """Enterprise-grade database persistence layer for Voltforge AI."""

    def __init__(self):
        self.host = config.DB_HOST
        self.port = config.DB_PORT
        self.user = config.DB_USER
        self.password = config.DB_PASSWORD
        self.database = config.DB_NAME
        self.enabled = config.DB_ENABLED
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="voltforge-db-worker")

    def get_connection(self) -> Optional[pymysql.Connection]:
        """Establish MySQL connection with timeout."""
        if not self.enabled:
            return None
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                conn = pymysql.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    cursorclass=pymysql.cursors.DictCursor,
                    connect_timeout=3,
                    read_timeout=5,
                    write_timeout=5,
                    charset="utf8mb4",
                    autocommit=False,
                )
                return conn
            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logger.warning(f"Database connection attempt {attempt+1} failed ({self.host}:{self.port}): {exc}")
                    return None
        return None

    @contextmanager
    def session_scope(self):
        """Transactional scope with auto-rollback on error and safe close."""
        conn = self.get_connection()
        if conn is None:
            yield None
            return
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error(f"DB Transaction error, rolled back: {exc}", exc_info=False)
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def run_async(self, func: Callable, *args, **kwargs) -> None:
        """Submits a database task asynchronously to avoid blocking HTTP threads."""
        try:
            self._executor.submit(func, *args, **kwargs)
        except Exception as exc:
            logger.warning(f"Failed to submit async DB task: {exc}")

    def check_health(self) -> Dict[str, Any]:
        """Check database reachability and table status."""
        if not self.enabled:
            return {"connected": False, "reason": "Database disabled in config"}
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return {"connected": False, "reason": "Could not establish connection"}
                with conn.cursor() as cur:
                    cur.execute("SHOW TABLES")
                    tables = [list(r.values())[0] for r in cur.fetchall()]
                    return {
                        "connected": True,
                        "database": self.database,
                        "host": self.host,
                        "tableCount": len(tables),
                        "tables": tables,
                    }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    # -------------------------------------------------------------------------
    # 1. Chat Sessions & Messages
    # -------------------------------------------------------------------------
    def save_chat_turn(
        self,
        session_id: Optional[str],
        project_id: Optional[str],
        user_id: Optional[str],
        user_message: str,
        assistant_reply: str,
        board_type: str = "ARDUINO_UNO",
        intent: str = "UNKNOWN",
        confidence: float = 0.85,
        generated_code: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        wire_suggestions: Optional[List[Dict[str, Any]]] = None,
        components_count: int = 0,
        wires_count: int = 0,
        tokens_input: int = 0,
        tokens_output: int = 0,
        latency_ms: float = 0.0,
        model_version: str = "VoltForge-1.0-Hybrid",
    ) -> Optional[str]:
        """Persist a user turn and assistant reply atomically."""
        if not self.enabled:
            return None

        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None

                with conn.cursor() as cur:
                    actual_session_id = session_id or str(uuid.uuid4())

                    # Ensure session exists in ai_sessions
                    cur.execute(
                        """
                        INSERT INTO ai_sessions (id, project_id, user_id, session_title, board_type, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON DUPLICATE KEY UPDATE
                            updated_at = CURRENT_TIMESTAMP,
                            board_type = VALUES(board_type)
                        """,
                        (
                            actual_session_id,
                            project_id,
                            user_id,
                            user_message[:100] if user_message else "New Session",
                            board_type,
                        ),
                    )

                    # Insert User Message
                    user_msg_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_chat_messages
                        (id, session_id, sender_role, message_text, tokens_prompt, latency_ms)
                        VALUES (%s, %s, 'USER', %s, %s, 0.0)
                        """,
                        (user_msg_id, actual_session_id, user_message, tokens_input),
                    )

                    # Insert Assistant Message
                    asst_msg_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_chat_messages
                        (id, session_id, sender_role, message_text, intent_detected, confidence_score,
                         generated_code, citations, suggested_actions,
                         tokens_completion, latency_ms)
                        VALUES (%s, %s, 'ASSISTANT', %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            asst_msg_id,
                            actual_session_id,
                            assistant_reply,
                            intent,
                            confidence,
                            generated_code,
                            safe_json_dumps(citations) if citations else None,
                            safe_json_dumps(wire_suggestions) if wire_suggestions else None,
                            tokens_output,
                            latency_ms,
                        ),
                    )

                    return actual_session_id
        except Exception as exc:
            logger.warning(f"Failed to save chat turn: {exc}")
            return None

    # -------------------------------------------------------------------------
    # 2. Circuit Validations & Issues
    # -------------------------------------------------------------------------
    def save_circuit_validation(
        self,
        project_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_type: str = "ARDUINO_UNO",
        is_valid: bool = True,
        safety_score: int = 100,
        general_feedback: str = "",
        component_count: int = 0,
        wire_count: int = 0,
        issues: Optional[List[Dict[str, Any]]] = None,
        suggested_additions: Optional[List[Dict[str, Any]]] = None,
        additions: Optional[List[Dict[str, Any]]] = None,
        removals: Optional[List[Dict[str, Any]]] = None,
        code_fixes: Optional[List[Dict[str, Any]]] = None,
        duration_ms: float = 0.0,
        execution_time_ms: float = 0.0,
    ) -> Optional[str]:
        """Save a circuit validation result and its individual issue breakdown."""
        if not self.enabled:
            return None

        issues = issues or []
        exec_time = duration_ms if duration_ms > 0 else execution_time_ms
        add_json = safe_json_dumps(additions or suggested_additions) if (additions or suggested_additions) else None
        rem_json = safe_json_dumps(removals) if removals else None
        cf_json = safe_json_dumps(code_fixes) if code_fixes else None

        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None

                with conn.cursor() as cur:
                    val_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_circuit_validations
                        (id, snapshot_id, project_id, is_valid, safety_score, general_feedback,
                         additions_json, removals_json, code_fixes_json, execution_time_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            val_id,
                            snapshot_id,
                            project_id,
                            is_valid,
                            safety_score,
                            general_feedback,
                            add_json,
                            rem_json,
                            cf_json,
                            exec_time,
                        ),
                    )

                    # Insert issues into ai_validation_issues if present
                    for iss in issues:
                        issue_id = str(uuid.uuid4())
                        raw_sev = str(iss.get("severity", "WARNING")).upper()
                        norm_sev = raw_sev if raw_sev in ("CRITICAL", "WARNING", "INFO", "SUGGESTION") else "WARNING"
                        cur.execute(
                            """
                            INSERT INTO ai_validation_issues
                            (id, validation_id, severity, category, component_id, pin_name, message, suggested_fix, is_resolved)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                            """,
                            (
                                issue_id,
                                val_id,
                                norm_sev,
                                iss.get("category", "ELECTRICAL_SAFETY"),
                                iss.get("componentId") or iss.get("component_id"),
                                iss.get("pinId") or iss.get("pin_id") or iss.get("pinName"),
                                iss.get("message", "Validation issue detected"),
                                iss.get("suggestedFix") or iss.get("suggested_fix"),
                            ),
                        )

                    return val_id
        except Exception as exc:
            logger.warning(f"Failed to save circuit validation: {exc}")
            return None

    # -------------------------------------------------------------------------
    # 3. Code Generation History
    # -------------------------------------------------------------------------
    def save_code_generation(
        self,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        board_type: str = "ARDUINO_UNO",
        target_mcu: Optional[str] = None,
        prompt: str = "",
        prompt_text: Optional[str] = None,
        generated_code: str = "",
        framework: str = "ARDUINO_CPP",
        generation_mode: str = "HYBRID_AST",
        libraries_used: Optional[List[str]] = None,
        pin_mappings: Optional[Dict[str, Any]] = None,
        compiles_cleanly: Optional[bool] = None,
        generation_time_ms: float = 0.0,
    ) -> Optional[str]:
        if not self.enabled:
            return None

        actual_prompt = prompt or prompt_text or "Code Generation"
        actual_board = board_type or target_mcu or "ARDUINO_UNO"

        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    code_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_code_generations
                        (id, session_id, project_id, board_type, prompt, generated_code, framework,
                         libraries_used_json, pin_mappings_json, compiles_cleanly, generation_time_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            code_id,
                            session_id,
                            project_id,
                            actual_board,
                            actual_prompt,
                            generated_code,
                            framework,
                            safe_json_dumps(libraries_used) if libraries_used else None,
                            safe_json_dumps(pin_mappings) if pin_mappings else None,
                            compiles_cleanly,
                            generation_time_ms,
                        ),
                    )
                    return code_id
        except Exception as exc:
            logger.warning(f"Failed to save code generation: {exc}")
            return None

    # -------------------------------------------------------------------------
    # 4. SPICE Simulation Cache
    # -------------------------------------------------------------------------
    def save_spice_simulation(
        self,
        circuit_hash: str,
        analysis_type: str,
        netlist_content: str,
        node_voltages: Optional[Dict[str, Any]] = None,
        branch_currents: Optional[Dict[str, Any]] = None,
        power_dissipation: Optional[Dict[str, Any]] = None,
        converged: bool = True,
        iterations: int = 1,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> bool:
        if not self.enabled:
            return False
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    sim_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_spice_simulations
                        (id, circuit_hash, analysis_type, netlist_content,
                         node_voltages_json, branch_currents_json, power_dissipation_json,
                         converged, iterations, solver_duration_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            node_voltages_json = VALUES(node_voltages_json),
                            branch_currents_json = VALUES(branch_currents_json),
                            power_dissipation_json = VALUES(power_dissipation_json),
                            converged = VALUES(converged),
                            iterations = VALUES(iterations),
                            solver_duration_ms = VALUES(solver_duration_ms)
                        """,
                        (
                            sim_id,
                            circuit_hash,
                            analysis_type,
                            netlist_content,
                            safe_json_dumps(node_voltages) if node_voltages else None,
                            safe_json_dumps(branch_currents) if branch_currents else None,
                            safe_json_dumps(power_dissipation) if power_dissipation else None,
                            converged,
                            iterations,
                            duration_ms,
                        ),
                    )
                    return True
        except Exception as exc:
            logger.warning(f"Failed to save SPICE simulation: {exc}")
            return False

    def get_cached_spice_simulation(self, circuit_hash: str, analysis_type: str = "OP_DC") -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT circuit_hash, analysis_type, node_voltages_json, branch_currents_json,
                               power_dissipation_json, converged, iterations, solver_duration_ms
                        FROM ai_spice_simulations
                        WHERE circuit_hash = %s AND analysis_type = %s
                        LIMIT 1
                        """,
                        (circuit_hash, analysis_type),
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "circuitHash": row["circuit_hash"],
                            "analysisType": row["analysis_type"],
                            "nodeVoltages": json.loads(row["node_voltages_json"]) if row["node_voltages_json"] else {},
                            "branchCurrents": json.loads(row["branch_currents_json"]) if row["branch_currents_json"] else {},
                            "powerDissipation": json.loads(row["power_dissipation_json"]) if row["power_dissipation_json"] else {},
                            "converged": bool(row["converged"]),
                            "durationMs": row["solver_duration_ms"],
                        }
        except Exception as exc:
            logger.warning(f"Failed to fetch cached SPICE result: {exc}")
        return None

    # -------------------------------------------------------------------------
    # 5. Prompt Rules & Active Rules Retrieval
    # -------------------------------------------------------------------------
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Retrieve all active prompt rules from ai_prompt_rules_and_templates."""
        if not self.enabled:
            return []
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT rule_code, category, title, description, severity_level, rule_logic_expression, suggested_remediation
                        FROM ai_prompt_rules_and_templates
                        WHERE is_active = TRUE
                        ORDER BY id ASC
                        """
                    )
                    return cur.fetchall()
        except Exception as exc:
            logger.warning(f"Failed to load active rules from DB: {exc}")
            return []

    def get_prompt_rules(self, board_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.get_active_rules()

    # -------------------------------------------------------------------------
    # 6. User Feedback
    # -------------------------------------------------------------------------
    def save_feedback(
        self,
        message_id: Optional[str],
        session_id: Optional[str],
        user_id: Optional[str],
        feedback_type: str,
        rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        corrected_code: Optional[str] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    fb_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_user_feedback
                        (id, message_id, user_id, rating, feedback_comment, user_corrected_code)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (fb_id, message_id, user_id, rating, feedback_text, corrected_code),
                    )
                    return fb_id
        except Exception as exc:
            logger.warning(f"Failed to record feedback: {exc}")
            return None


# Global Singleton Instance
db = DatabaseManager()
