import zlib
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

def deterministic_tokenize(text: str, max_len: int = 128, vocab_size: int = 32000) -> Tuple[List[int], List[int]]:
    """
    Deterministic CRC32-based tokenization ensuring identical token IDs across processes.
    """
    cleaned = text.lower().replace("{", " ").replace("}", " ").replace(":", " ").replace('"', " ").replace(",", " ")
    words = cleaned.split()
    tokens = [101]  # [CLS]
    for w in words:
        tid = 100 + (zlib.crc32(w.encode("utf-8")) % (vocab_size - 200))
        tokens.append(tid)
    tokens.append(102)  # [SEP]
    
    orig_len = len(tokens)
    if orig_len < max_len:
        pad_len = max_len - orig_len
        padded_tokens = tokens + [0] * pad_len
        mask = [1] * orig_len + [0] * pad_len
    else:
        padded_tokens = tokens[:max_len]
        mask = [1] * max_len
        
    return padded_tokens, mask


# Domain templates mapped to action indices
# 0: NOOP, 1: EXECUTE_WORKFLOW, 2: ROUTE_DATABASE, 3: ROUTE_AUTH,
# 4: ROUTE_PAYMENT, 5: TRIGGER_ALERT, 7: APPLY_RATE_LIMIT
DOMAIN_TEMPLATES = [
    # 0: NOOP (Healthcheck / Ping / Idle)
    (
        0, 0.05, 0,
        [
            "service healthcheck event heartbeat status ok ping alive normal",
            "service monitor event ping idle system nominal no operation needed",
            "service liveness event keepalive ok check health routine tick"
        ]
    ),
    # 2: ROUTE_DATABASE (Queries, Shards, Replicas)
    (
        2, 0.20, 2,
        [
            "service databasecoordinator event read_replica_sync priority normal table orders target_shard records query sql",
            "service postgres_cluster event query_execution read write table users shard transaction connection pool",
            "service data_mesh event replicate_records table payments partition migrate database storage"
        ]
    ),
    # 3: ROUTE_AUTH (Login, JWT, OAuth, MFA)
    (
        3, 0.15, 1,
        [
            "service authservice event user_login_mfa priority normal user_id auth_method fido2 webauthn jwt verify",
            "service identity_provider event validate_token session oauth credentials permissions access security",
            "service sso_gateway event refresh_token user authenticate grant permissions profile"
        ]
    ),
    # 4: ROUTE_PAYMENT (Billing, Card, Stripe, Checkout)
    (
        4, 0.85, 5,
        [
            "service paymentprocessor event incoming_charge priority high amount currency usd card_brand visa checkout",
            "service billing_engine event capture_funds credit_card stripe transaction invoice settle payment",
            "service checkout_worker event process_payment charge customer wallet gateway pci authorize"
        ]
    ),
    # 5: TRIGGER_ALERT (Outages, Panics, Failures)
    (
        5, 0.95, 9,
        [
            "service incidentmanager event critical_system_failure priority critical panic alert pagerduty outage emergency",
            "service cluster_monitor event memory_exhaustion node_down severity critical trigger oncall alarm",
            "service kernel_watchdog event fatal_crash subsystem deadlock emergency alert ops team"
        ]
    ),
    # 7: APPLY_RATE_LIMIT (Suspicious Traffic, Abuse, Botnet)
    (
        7, 0.90, 8,
        [
            "service api_gateway event suspicious_traffic priority critical ip_address request_rate req_sec flood ddos attack",
            "service ingress_controller event rate_limit_exceeded burst abusive botnet throttle ip ban 429",
            "service waf_shield event high_frequency_traffic suspicious client anomaly block rate limit"
        ]
    )
]

class JustActionSyntheticDataset(Dataset):
    """
    Realistic orchestration dataset generating structured token sequences
    mapped to distinct domain actions, noul boolean targets, and rubric scores.
    """
    def __init__(self, samples: int = 1500, seq_len: int = 128, vocab_size: int = 32000):
        all_ids = []
        all_masks = []
        all_choices = []
        all_nouls = []
        all_scores = []

        random.seed(42)

        for _ in range(samples):
            # Pick a domain template
            action_idx, noul_target, score_target, phrases = random.choice(DOMAIN_TEMPLATES)
            base_text = random.choice(phrases)

            # Add minor perturbations (jitter tokens) for generalization
            words = base_text.split()
            if random.random() < 0.3 and len(words) > 4:
                random.shuffle(words)
            text = " ".join(words)

            tokens, mask = deterministic_tokenize(text, max_len=seq_len, vocab_size=vocab_size)

            all_ids.append(tokens)
            all_masks.append(mask)
            all_choices.append(action_idx)
            # Add minor Gaussian noise to continuous score and noul targets
            all_nouls.append(min(max(noul_target + random.uniform(-0.02, 0.02), 0.0), 1.0))
            all_scores.append(score_target)

        self.input_ids = torch.tensor(all_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(all_masks, dtype=torch.long)
        self.target_choice = torch.tensor(all_choices, dtype=torch.long)
        self.target_noul = torch.tensor(all_nouls, dtype=torch.float)
        self.target_score = torch.tensor(all_scores, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "target_choice": self.target_choice[idx],
            "target_noul": self.target_noul[idx],
            "target_score": self.target_score[idx]
        }

# Alias for backward compatibility
JevSyntheticDataset = JustActionSyntheticDataset