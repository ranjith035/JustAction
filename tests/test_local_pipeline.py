import unittest
import os
import sys
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model import JustActionConfig, JustActionModel
from dataset import JustActionSyntheticDataset
from optimize_local import ExportWrapper

class TestJustActionPipeline(unittest.TestCase):
    def setUp(self):
        self.config = JustActionConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            max_choices=5,
            max_score_levels=5
        )
        self.model = JustActionModel(self.config)
        self.model.eval()

    def test_model_forward(self):
        input_ids = torch.randint(0, 1000, (2, 32))
        mask = torch.ones((2, 32), dtype=torch.long)
        
        # Test dict return
        outputs = self.model(input_ids, attention_mask=mask)
        self.assertIn("choice_probs", outputs)
        self.assertIn("noul_prob", outputs)
        self.assertIn("score_dist", outputs)
        
        self.assertEqual(outputs["choice_probs"].shape, (2, 5))
        self.assertEqual(outputs["noul_prob"].shape, (2,))
        self.assertEqual(outputs["score_dist"].shape, (2, 5))

    def test_export_wrapper(self):
        wrapper = ExportWrapper(self.model)
        input_ids = torch.randint(0, 1000, (1, 16))
        mask = torch.ones((1, 16), dtype=torch.long)
        
        choice_probs, noul_prob, score_dist = wrapper(input_ids, mask)
        self.assertEqual(choice_probs.shape, (1, 5))
        self.assertEqual(noul_prob.shape, (1,))
        self.assertEqual(score_dist.shape, (1, 5))

    def test_dataset_generation(self):
        ds = JustActionSyntheticDataset(samples=20, seq_len=16, vocab_size=1000)
        self.assertEqual(len(ds), 20)
        sample = ds[0]
        self.assertIn("input_ids", sample)
        self.assertIn("target_choice", sample)
        self.assertIn("target_noul", sample)
        self.assertIn("target_score", sample)

if __name__ == "__main__":
    unittest.main()
