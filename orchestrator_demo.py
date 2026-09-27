"""
JustAction System 1 Orchestrator Demo
-------------------------------------
Demonstrates JustAction as an ultra-fast (<10ms) reflex decision engine
inside an event-driven workflow orchestrator.

Workflow:
  Incoming Event -> Orchestrator parses Context -> JustAction System 1 Model
                 -> Typed Decision (Action, Confidence, Noul, Score)
                 -> Orchestrator executes Action Handler -> State Updated
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from local_runner import LocalJustActionEngine
from dataset import deterministic_tokenize

@dataclass
class OrchestrationEvent:
    task_id: str
    service: str
    event_type: str
    payload: Dict[str, Any]
    retry_count: int = 0
    priority: str = "NORMAL"

    def to_input_tokens(self) -> List[int]:
        """
        Encodes orchestration event metadata into deterministic integer token representations.
        """
        raw_text = f"service {self.service} event {self.event_type} priority {self.priority} {json.dumps(self.payload)}"
        tokens, _ = deterministic_tokenize(raw_text, max_len=128)
        return tokens


class JustActionOrchestrator:
    """
    Event-driven Orchestrator powered by JustAction INT8 System 1 Engine.
    """
    def __init__(self, model_path: str = "justaction_int8.onnx", confidence_threshold: float = 0.70):
        self.engine = LocalJustActionEngine(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            verbose=False
        )
        self.execution_history: List[Dict[str, Any]] = []

        # Action Handler Dispatch Table
        self.handlers = {
            "NOOP": self._handle_noop,
            "EXECUTE_WORKFLOW": self._handle_execute_workflow,
            "ROUTE_DATABASE": self._handle_route_database,
            "ROUTE_AUTH": self._handle_route_auth,
            "ROUTE_PAYMENT": self._handle_route_payment,
            "TRIGGER_ALERT": self._handle_trigger_alert,
            "ESCALATE_TIER2": self._handle_escalate_tier2,
            "APPLY_RATE_LIMIT": self._handle_apply_rate_limit,
            "ARCHIVE_RECORD": self._handle_archive_record,
            "RETRY_OPERATION": self._handle_retry_operation,
            "FALLBACK_TO_HUMAN": self._handle_fallback_to_human
        }

    # ==========================================
    # Action Handlers Executed by Orchestrator
    # ==========================================

    def _handle_route_database(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        shard = params.get("target_shard", 1)
        return f"[DB_HANDLER] Routed query to Database Cluster Shard #{shard} (connection pooled)."

    def _handle_route_payment(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        amount = event.payload.get("amount", 0)
        return f"[PAYMENT_HANDLER] Dispatched transaction ${amount} to PCI-DSS secure gateway."

    def _handle_route_auth(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        user = event.payload.get("user_id", "anonymous")
        return f"[AUTH_HANDLER] Verified JWT session tokens for user '{user}'."

    def _handle_trigger_alert(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        severity = params.get("severity", "CRITICAL")
        return f"[ALERT_HANDLER] Dispatched high-priority PagerDuty / Slack alert (Level: {severity})."

    def _handle_apply_rate_limit(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        ip = event.payload.get("ip_address", "0.0.0.0")
        return f"[SECURITY_HANDLER] Applied Leaky Bucket Rate-Limit to IP {ip} (429 Too Many Requests)."

    def _handle_retry_operation(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        return f"[RETRY_HANDLER] Task rescheduled with exponential backoff (attempt {event.retry_count + 1})."

    def _handle_execute_workflow(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        return f"[WORKFLOW_HANDLER] Initialized DAG pipeline for '{event.service}'."

    def _handle_escalate_tier2(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        return f"[ESCALATION_HANDLER] Escalated task to Tier-2 Engineering on-call pool."

    def _handle_archive_record(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        return f"[STORAGE_HANDLER] Cold-stored event snapshot into S3 Glacier storage tier."

    def _handle_noop(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        return "[NOOP_HANDLER] No operation required. Event discarded safely."

    def _handle_fallback_to_human(self, event: OrchestrationEvent, params: Dict[str, Any]) -> str:
        reason = params.get("reason", "Confidence score below safety threshold")
        return f"[HUMAN_FALLBACK] Low model certainty ({reason}). Task quarantined for human supervisor review."

    # ==========================================
    # Main Orchestration Loop
    # ==========================================

    def dispatch(self, event: OrchestrationEvent) -> Dict[str, Any]:
        """
        Orchestration Cycle:
        1. Encodes event into input sequence
        2. Evaluates System 1 JustAction model in <10ms
        3. Invokes corresponding typed action handler
        4. Logs and returns execution receipt
        """
        t0 = time.perf_counter()
        
        # 1. Feature Representation
        input_ids = event.to_input_tokens()
        
        # 2. Fast System 1 Inference (< 10ms)
        raw_output = self.engine.predict_raw(input_ids)
        decision = self.engine.step(input_ids, parameters=event.payload)
        
        latency_ms = (time.perf_counter() - t0) * 1000

        action = decision["action"]
        confidence = decision["confidence"]
        fallback_triggered = decision["fallback_triggered"]

        # 3. Dynamic Override Based on Orchestration Rules
        # If security alert or high rubric score detected, prioritize safety
        if raw_output["expected_score"] > 8.0 and not fallback_triggered:
            action = "TRIGGER_ALERT"
            decision["action"] = action

        # 4. Dispatch Action Handler
        handler = self.handlers.get(action, self._handle_fallback_to_human)
        handler_result = handler(event, decision["parameters"])

        receipt = {
            "task_id": event.task_id,
            "service": event.service,
            "decision": decision,
            "action_executed": action,
            "confidence": confidence,
            "noul_prob": raw_output["noul_probability"],
            "rubric_score": raw_output["expected_score"],
            "latency_ms": round(latency_ms, 2),
            "handler_message": handler_result
        }

        self.execution_history.append(receipt)
        return receipt


def run_orchestrator_scenarios(threshold: float, mode_title: str):
    print("=" * 70)
    print(f" {mode_title} (Threshold = {threshold})")
    print("=" * 70)

    orchestrator = JustActionOrchestrator(
        model_path="justaction_int8.onnx",
        confidence_threshold=threshold
    )

    # Sample Orchestration Events from different enterprise systems
    events = [
        OrchestrationEvent(
            task_id="TASK-101",
            service="PaymentProcessor",
            event_type="INCOMING_CHARGE",
            payload={"amount": 420.50, "currency": "USD", "card_brand": "Visa"},
            priority="HIGH"
        ),
        OrchestrationEvent(
            task_id="TASK-102",
            service="DatabaseCoordinator",
            event_type="READ_REPLICA_SYNC",
            payload={"table": "orders", "target_shard": 3, "records": 12000},
            priority="NORMAL"
        ),
        OrchestrationEvent(
            task_id="TASK-103",
            service="API_Gateway",
            event_type="SUSPICIOUS_TRAFFIC",
            payload={"ip_address": "198.51.100.42", "request_rate": "850 req/sec"},
            priority="CRITICAL"
        )
    ]

    for i, event in enumerate(events, start=1):
        print("-" * 70)
        print(f"[Event #{i}] ID: {event.task_id} | Service: {event.service}")
        print(f"  Event Type : {event.event_type} (Priority: {event.priority})")
        print(f"  Parameters : {json.dumps(event.payload)}")

        # Orchestrator handles the event using JustAction System 1
        receipt = orchestrator.dispatch(event)

        print("\n  [JustAction System 1 Decision]:")
        print(f"    - Action Decision   : {receipt['action_executed']}")
        print(f"    - Calibrated Conf   : {receipt['confidence']:.4f}")
        print(f"    - Fallback Active   : {receipt['decision']['fallback_triggered']}")
        print(f"    - Decision Latency  : {receipt['latency_ms']} ms")

        print("\n  [Orchestrator Execution Handler]:")
        print(f"    -> {receipt['handler_message']}")
        print()


def main():
    print("\n" + "#" * 70)
    print("       JUSTACTION SYSTEM 1 - WORKFLOW ORCHESTRATOR DEMO")
    print("#" * 70 + "\n")

    # Mode A: Automated Direct Action Routing
    run_orchestrator_scenarios(
        threshold=0.20,
        mode_title="MODE A: HIGH-SPEED AUTOMATED ACTION DISPATCH"
    )

    # Mode B: Strict Safety Fallback (Quarantine low certainty to Human)
    run_orchestrator_scenarios(
        threshold=0.70,
        mode_title="MODE B: STRICT SAFETY GUARDRAIL (FALLBACK TO HUMAN)"
    )

    print("#" * 70)
    print("Orchestration pipeline execution complete. Zero token-loop latency.")
    print("#" * 70 + "\n")

if __name__ == "__main__":
    main()
