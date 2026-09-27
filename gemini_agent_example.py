"""
JustAction + Google Gemini: Dual-Process Cognitive Architecture
---------------------------------------------------------------
Demonstrates pairing JustAction (System 1 Fast Reflex, <10ms, $0 cost)
with Google Gemini (System 2 Deep Reasoning & Planning).

Workflow:
1. Incoming task is evaluated by JustAction in a single parallel pass (<10ms).
2. If Confidence >= 0.70: Action executed immediately (zero LLM token cost).
3. If Confidence < 0.70: Task is escalated to Google Gemini for deep reasoning.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from local_runner import LocalJustActionEngine
from dataset import deterministic_tokenize

class GeminiHybridAgent:
    """
    Dual-process agent combining JustAction (System 1) and Google Gemini (System 2).
    """
    def __init__(
        self,
        model_path: str = "justaction_int8.onnx",
        confidence_threshold: float = 0.70,
        gemini_model: str = "gemini-2.5-flash"
    ):
        # 1. Initialize local System 1 engine
        self.system1 = LocalJustActionEngine(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            verbose=False
        )
        self.gemini_model = gemini_model
        self.api_key = os.environ.get("GEMINI_API_KEY")

        # 2. Initialize Gemini client if API key is provided
        self.gemini_client = None
        if self.api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key)
            except ImportError:
                pass

    def run_step(self, event_text: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a dual-process step:
        - Evaluates System 1 reflex (<10ms).
        - Escalates to Gemini only if confidence is below threshold.
        """
        params = parameters or {}
        print("-" * 65)
        print(f"[INCOMING EVENT] '{event_text}'")
        print(f" Parameters: {json.dumps(params)}")

        t0 = time.perf_counter()
        
        # Tokenize event into deterministic feature representation
        tokens, mask = deterministic_tokenize(event_text, max_len=128)
        
        # -------------------------------------------------------------
        # Phase 1: System 1 (JustAction) - Latency: <10ms, Cost: $0.00
        # -------------------------------------------------------------
        decision = self.system1.step(tokens, attention_mask=mask, parameters=params)
        raw_output = self.system1.predict_raw(tokens, attention_mask=mask)
        system1_latency = (time.perf_counter() - t0) * 1000

        confidence = decision["confidence"]
        action = decision["action"]
        fallback = decision["fallback_triggered"]

        print(f"\n  [Phase 1: JustAction System 1 Reflex]")
        print(f"    - Action Decision   : {action}")
        print(f"    - Calibrated Conf   : {confidence:.4f}")
        print(f"    - Noul Probability  : {raw_output['noul_probability']:.4f}")
        print(f"    - Execution Latency : {system1_latency:.2f} ms")

        # High confidence -> Instant execution without calling Gemini
        if not fallback:
            print(f"    -> [SUCCESS] Handled by System 1 in {system1_latency:.2f}ms! (Saved Gemini API call)")
            return {
                "executor": "SYSTEM_1_JUSTACTION",
                "action": action,
                "confidence": confidence,
                "parameters": params,
                "latency_ms": round(system1_latency, 2),
                "cost": "$0.00 (Local Execution)"
            }

        # -------------------------------------------------------------
        # Phase 2: System 2 (Google Gemini) - Deep Reasoning Fallback
        # -------------------------------------------------------------
        print(f"\n  [Phase 2: Escalating to System 2 - Google Gemini]")
        print(f"    - Reason: Confidence ({confidence:.4f}) below safety threshold.")
        
        t_gemini_start = time.perf_counter()
        gemini_reasoning = self._call_gemini(event_text, params)
        gemini_latency = (time.perf_counter() - t_gemini_start) * 1000

        print(f"    - Gemini Latency    : {gemini_latency:.2f} ms")
        print(f"    -> [COMPLETED] Deep reasoning completed by Gemini.")

        return {
            "executor": "SYSTEM_2_GEMINI",
            "action": "DELEGATED_TO_GEMINI",
            "confidence": 1.0,
            "gemini_output": gemini_reasoning,
            "system1_latency_ms": round(system1_latency, 2),
            "total_latency_ms": round(system1_latency + gemini_latency, 2),
            "cost": "Standard Gemini API Tokens"
        }

    def _call_gemini(self, event_text: str, parameters: Dict[str, Any]) -> str:
        """Invokes Gemini API or provides simulated response if API key is not set."""
        prompt = (
            f"You are an AI supervisor for an automated system.\n"
            f"The System 1 reflex unit was uncertain about this request:\n"
            f"Request: {event_text}\n"
            f"Parameters: {json.dumps(parameters)}\n"
            f"Analyze the request, decide the proper action, and explain your reasoning."
        )

        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                return f"[Gemini API Call Failed: {e}]"
        else:
            return (
                f"[SIMULATED GEMINI RESPONSE - Set GEMINI_API_KEY to call live API]\n"
                f"Analysis: The incoming request is non-standard and requires multi-step escalation.\n"
                f"Decision: Dispatching custom enterprise verification workflow and notifying on-call lead."
            )


def main():
    print("=" * 65)
    print("  JUSTACTION + GOOGLE GEMINI: DUAL-PROCESS AGENT DEMO")
    print("=" * 65)
    print("Architecture:")
    print("  1. System 1 (JustAction INT8): <10ms local reflex, $0 cost")
    print("  2. System 2 (Google Gemini)  : Deep reasoning when confidence < 0.70\n")

    # Initialize agent (works with live Gemini API or simulation mode)
    agent = GeminiHybridAgent(
        model_path="justaction_int8.onnx",
        confidence_threshold=0.70
    )

    # Scenario 1: Routine task -> JustAction reflex handles it immediately
    print("\n>>> TEST SCENARIO 1: Routine Billing Operation")
    res1 = agent.run_step(
        event_text="service paymentprocessor incoming_charge amount 350 card_brand mastercard currency usd",
        parameters={"amount": 350.00, "currency": "USD"}
    )
    print(f"\nExecution Receipt 1:\n{json.dumps(res1, indent=2)}")

    # Scenario 2: Novel / Ambiguous request -> Escalates to Gemini System 2
    print("\n>>> TEST SCENARIO 2: Complex / Unfamiliar Request")
    res2 = agent.run_step(
        event_text="Please negotiate an extended SLA penalty clause and audit data residency compliance.",
        parameters={"customer": "Enterprise_Bank_Corp", "risk_level": "UNKNOWN"}
    )
    print(f"\nExecution Receipt 2:\n{json.dumps(res2, indent=2)}")

    print("\n" + "=" * 65)
    print("Dual-Process demonstration complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()
