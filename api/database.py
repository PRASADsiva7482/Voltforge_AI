import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pymysql
import pymysql.cursors
import config

logger = logging.getLogger("voltforge-ai.database")


class DatabaseManager:
    """Database persistence layer for Voltforge AI microservice."""

    def __init__(self):
        self.host = config.DB_HOST
        self.port = config.DB_PORT
        self.user = config.DB_USER
        self.password = config.DB_PASSWORD
        self.database = config.DB_NAME
        self.enabled = config.DB_ENABLED

    def get_connection(self) -> Optional[pymysql.Connection]:
        if not self.enabled:
            return None
        try:
            return pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=3,
                charset="utf8mb4",
                autocommit=False,
            )
        except Exception as exc:
            logger.warning(f"Failed to connect to Voltforge_AI database ({self.host}:{self.port}): {exc}")
            return None

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        conn = self.get_connection()
        if conn is None:
            yield None
            return
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"DB Transaction error, rolled back: {exc}", exc_info=True)
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

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
        suggested_actions: Optional[Dict[str, Any]] = None,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: int = 0,
    ) -> Optional[str]:
        """Persist a multi-turn chat interaction to ai_sessions and ai_chat_messages."""
        if not self.enabled:
            return None
        session_id = session_id or str(uuid.uuid4())
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    # 1. Upsert session
                    cur.execute(
                        """
                        INSERT INTO ai_sessions (id, project_id, user_id, session_title, board_type, turn_count, status)
                        VALUES (%s, %s, %s, %s, %s, 1, 'ACTIVE')
                        ON DUPLICATE KEY UPDATE
                            turn_count = turn_count + 1,
                            board_type = VALUES(board_type),
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (session_id, project_id, user_id, user_message[:50] or "Electronics Chat", board_type),
                    )

                    # 2. Insert user message
                    user_msg_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_chat_messages
                        (id, session_id, sender_role, message_text, intent_detected, confidence_score, created_at)
                        VALUES (%s, %s, 'USER', %s, %s, 1.0, CURRENT_TIMESTAMP)
                        """,
                        (user_msg_id, session_id, user_message, intent),
                    )

                    # 3. Insert assistant reply
                    asst_msg_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO ai_chat_messages
                        (id, session_id, sender_role, message_text, intent_detected, confidence_score,
                         generated_code, citations, suggested_actions, tokens_prompt, tokens_completion, latency_ms, created_at)
                        VALUES (%s, %s, 'ASSISTANT', %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """,
                        (
                            asst_msg_id,
                            session_id,
                            assistant_reply,
                            intent,
                            float(confidence),
                            generated_code,
                            json.dumps(citations or []),
                            json.dumps(suggested_actions or {}),
                            tokens_prompt,
                            tokens_completion,
                            latency_ms,
                        ),
                    )
            return session_id
        except Exception as exc:
            logger.warning(f"Error persisting chat turn: {exc}")
            return None

    # -------------------------------------------------------------------------
    # 2. Canvas Snapshots
    # -------------------------------------------------------------------------
    def save_canvas_snapshot(
        self,
        project_id: str,
        board_type: str,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        code: str = "",
        diagnostics: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save active circuit topology and editor snapshot."""
        if not self.enabled:
            return None
        snapshot_id = str(uuid.uuid4())
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ai_canvas_snapshots
                        (id, project_id, board_type, component_count, wire_count, components_json, wires_json, active_code_snapshot, compiler_diagnostics)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            snapshot_id,
                            project_id,
                            board_type,
                            len(components),
                            len(wires),
                            json.dumps(components),
                            json.dumps(wires),
                            code,
                            json.dumps(diagnostics or []),
                        ),
                    )
            return snapshot_id
        except Exception as exc:
            logger.warning(f"Error saving canvas snapshot: {exc}")
            return None

    # -------------------------------------------------------------------------
    # 3. Circuit Validations & Issues
    # -------------------------------------------------------------------------
    def save_circuit_validation(
        self,
        project_id: str,
        snapshot_id: Optional[str],
        is_valid: bool,
        safety_score: int,
        general_feedback: str,
        issues: List[Dict[str, Any]],
        additions: Optional[List[Dict[str, Any]]] = None,
        removals: Optional[List[Dict[str, Any]]] = None,
        code_fixes: Optional[List[Dict[str, Any]]] = None,
        duration_ms: int = 0,
    ) -> Optional[str]:
        """Record circuit audit results and all individual safety issues."""
        if not self.enabled:
            return None
        validation_id = str(uuid.uuid4())
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ai_circuit_validations
                        (id, snapshot_id, project_id, is_valid, safety_score, general_feedback, additions_json, removals_json, code_fixes_json, execution_time_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            validation_id,
                            snapshot_id,
                            project_id,
                            1 if is_valid else 0,
                            safety_score,
                            general_feedback,
                            json.dumps(additions or []),
                            json.dumps(removals or []),
                            json.dumps(code_fixes or []),
                            duration_ms,
                        ),
                    )

                    for issue in issues:
                        issue_id = str(uuid.uuid4())
                        raw_sev = str(issue.get("severity", "WARNING")).upper()
                        if raw_sev in ("ERROR", "DANGER", "CRITICAL"):
                            severity = "CRITICAL"
                        elif raw_sev in ("INFO", "INFORMATIONAL"):
                            severity = "INFO"
                        elif raw_sev in ("SUGGESTION", "FIX"):
                            severity = "SUGGESTION"
                        else:
                            severity = "WARNING"

                        cur.execute(
                            """
                            INSERT INTO ai_validation_issues
                            (id, validation_id, severity, category, component_id, pin_name, message, suggested_fix)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                issue_id,
                                validation_id,
                                severity,
                                str(issue.get("category", "CIRCUIT_CHECK")),
                                issue.get("componentId"),
                                issue.get("pinName"),
                                str(issue.get("message", "")),
                                str(issue.get("suggestedFix", "")),
                            ),
                        )
            return validation_id
        except Exception as exc:
            logger.warning(f"Error saving circuit validation: {exc}")
            return None

    # -------------------------------------------------------------------------
    # 4. Code Generation & Repair History
    # -------------------------------------------------------------------------
    def save_code_generation(
        self,
        project_id: str,
        session_id: Optional[str],
        target_mcu: str,
        prompt_text: str,
        generation_mode: str,
        generated_code: str,
        included_libraries: Optional[List[str]] = None,
        pin_assignments: Optional[Dict[str, Any]] = None,
        compilation_passed: Optional[bool] = None,
    ) -> Optional[str]:
        """Save firmware synthesis run to ai_code_generations."""
        if not self.enabled:
            return None
        gen_id = str(uuid.uuid4())
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ai_code_generations
                        (id, session_id, project_id, target_mcu, prompt_text, generation_mode, generated_code, included_libraries, pin_assignments, compilation_passed)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            gen_id,
                            session_id,
                            project_id,
                            target_mcu,
                            prompt_text,
                            generation_mode,
                            generated_code,
                            json.dumps(included_libraries or []),
                            json.dumps(pin_assignments or {}),
                            compilation_passed,
                        ),
                    )
            return gen_id
        except Exception as exc:
            logger.warning(f"Error saving code generation: {exc}")
            return None

    # -------------------------------------------------------------------------
    # 5. SPICE Simulation Cache
    # -------------------------------------------------------------------------
    def get_cached_spice_simulation(self, circuit_hash: str, analysis_type: str = "OP_DC") -> Optional[Dict[str, Any]]:
        """Retrieve precomputed SPICE simulation results if circuit netlist hash matches."""
        if not self.enabled:
            return None
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT node_voltages_json, branch_currents_json, power_dissipation_json, converged
                        FROM ai_spice_simulations
                        WHERE circuit_hash = %s AND analysis_type = %s
                        LIMIT 1
                        """,
                        (circuit_hash, analysis_type),
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "nodeVoltages": json.loads(row["node_voltages_json"]),
                            "branchCurrents": json.loads(row["branch_currents_json"]),
                            "powerDissipation": json.loads(row["power_dissipation_json"] or "{}"),
                            "converged": bool(row["converged"]),
                            "cached": True,
                        }
        except Exception as exc:
            logger.warning(f"Error fetching SPICE cache: {exc}")
        return None

    def save_spice_simulation(
        self,
        circuit_hash: str,
        analysis_type: str,
        netlist_content: str,
        node_voltages: Dict[str, Any],
        branch_currents: Dict[str, Any],
        power_dissipation: Optional[Dict[str, Any]] = None,
        converged: bool = True,
        duration_ms: int = 0,
    ) -> bool:
        """Cache SPICE simulation results."""
        if not self.enabled:
            return False
        sim_id = str(uuid.uuid4())
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ai_spice_simulations
                        (id, circuit_hash, analysis_type, netlist_content, node_voltages_json, branch_currents_json, power_dissipation_json, converged, solver_duration_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            node_voltages_json = VALUES(node_voltages_json),
                            branch_currents_json = VALUES(branch_currents_json),
                            converged = VALUES(converged),
                            solver_duration_ms = VALUES(solver_duration_ms)
                        """,
                        (
                            sim_id,
                            circuit_hash,
                            analysis_type,
                            netlist_content,
                            json.dumps(node_voltages),
                            json.dumps(branch_currents),
                            json.dumps(power_dissipation or {}),
                            1 if converged else 0,
                            duration_ms,
                        ),
                    )
            return True
        except Exception as exc:
            logger.warning(f"Error caching SPICE simulation: {exc}")
            return False

    # -------------------------------------------------------------------------
    # 6. RAG Knowledge & Prompt Rules Retrieval
    # -------------------------------------------------------------------------
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Fetch active safety and pin restriction rules from ai_prompt_rules_and_templates."""
        if not self.enabled:
            return []
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT rule_code, category, title, description, severity_level, suggested_remediation
                        FROM ai_prompt_rules_and_templates
                        WHERE is_active = 1
                        """
                    )
                    return cur.fetchall()
        except Exception as exc:
            logger.warning(f"Error fetching active rules from DB: {exc}")
            return []

    # -------------------------------------------------------------------------
    # 7. Telemetry Logging
    # -------------------------------------------------------------------------
    def log_telemetry(
        self,
        endpoint: str,
        client_ip: Optional[str] = None,
        http_status: int = 200,
        tokens_consumed: int = 0,
        duration_ms: int = 0,
        engine: str = "HYBRID_ENGINE",
        error_trace: Optional[str] = None,
    ) -> None:
        """Log system audit telemetry to ai_telemetry_audit_logs."""
        if not self.enabled:
            return
        log_id = str(uuid.uuid4())
        try:
            with self.session_scope() as conn:
                if conn is None:
                    return
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ai_telemetry_audit_logs
                        (id, endpoint, client_ip, http_status, tokens_consumed, total_duration_ms, inference_engine_used, error_trace)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (log_id, endpoint, client_ip, http_status, tokens_consumed, duration_ms, engine, error_trace),
                    )
        except Exception as exc:
            logger.debug(f"Telemetry logging silently skipped: {exc}")


# Global database manager singleton instance
db = DatabaseManager()
