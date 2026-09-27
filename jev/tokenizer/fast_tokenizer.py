import re
import json
from typing import List, Dict, Union, Optional
import torch

class FastTokenizer:
    """
    Lightweight, self-contained subword/word tokenizer designed for ultra-low latency
    System 1 decision pipelines. No external network requests or dependencies required.
    """
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_vocab_size: int = 4096):
        self.max_vocab_size = max_vocab_size
        if vocab is not None:
            self.vocab = vocab
            self.inv_vocab = {idx: token for token, idx in vocab.items()}
        else:
            self.vocab = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.CLS_TOKEN: 2,
                self.SEP_TOKEN: 3,
            }
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _tokenize_text(self, text: str) -> List[str]:
        # Clean and split into alphanumeric tokens & common punctuation
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def build_vocab(self, texts: List[str]):
        """
        Builds vocabulary from a list of strings, sorting by frequency.
        """
        freqs: Dict[str, int] = {}
        for text in texts:
            for token in self._tokenize_text(text):
                freqs[token] = freqs.get(token, 0) + 1

        # Sort by frequency descending
        sorted_tokens = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        
        idx = len(self.vocab)
        for token, _ in sorted_tokens:
            if idx >= self.max_vocab_size:
                break
            if token not in self.vocab:
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
                idx += 1

    def encode(self, text: str, max_length: int = 128) -> List[int]:
        tokens = self._tokenize_text(text)
        token_ids = [self.vocab[self.CLS_TOKEN]]
        
        for t in tokens[: max_length - 2]:
            token_ids.append(self.vocab.get(t, self.vocab[self.UNK_TOKEN]))
            
        token_ids.append(self.vocab[self.SEP_TOKEN])
        return token_ids

    def batch_encode(
        self, 
        texts: List[str], 
        max_length: int = 128, 
        device: Optional[torch.device] = None
    ) -> Tuple_Tensors:
        batch_ids = []
        batch_masks = []

        for text in texts:
            ids = self.encode(text, max_length=max_length)
            seq_len = len(ids)
            pad_len = max_length - seq_len
            
            padded_ids = ids + [self.vocab[self.PAD_TOKEN]] * pad_len
            mask = [1] * seq_len + [0] * pad_len

            batch_ids.append(padded_ids)
            batch_masks.append(mask)

        ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(batch_masks, dtype=torch.long, device=device)
        return ids_tensor, mask_tensor

    def save_vocab(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load_vocab(cls, filepath: str) -> "FastTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)


Tuple_Tensors = tuple[torch.Tensor, torch.Tensor]
