
import json, torch
from torch.utils.data import Dataset
class CausalBlocks(Dataset):
    def __init__(self, jsonl_path, limit_samples=None):
        self.rows=[]
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
                    if limit_samples and len(self.rows)>=int(limit_samples): break
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r=self.rows[idx]
        return {k:torch.tensor(r[k], dtype=torch.long) for k in ("input_ids","attention_mask","labels")}
def make_causal_dataset(jsonl_path, limit_samples=None):
    return CausalBlocks(jsonl_path, limit_samples)
