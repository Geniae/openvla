import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset


def load_samples_from_pt(path: Path) -> List[Dict[str, Any]]:
    """
    Returns a flat list of step dicts:
      {
        "image_primary": uint8[H,W,3],
        "action": float32[7],
        "language_instruction": str,
        # optional: "proprio": float32[d]
      }
    """
    data = torch.load(path, map_location="cpu")

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{path} must contain a non-empty list. Got: {type(data)} len={len(data) if isinstance(data, list) else 'n/a'}")

    # Case A: List[dict]
    if isinstance(data[0], dict):
        steps = data

    # Case B: List[List[dict]] -> flatten
    elif isinstance(data[0], list):
        steps = []
        for ep in data:
            if not ep:
                continue
            if not isinstance(ep[0], dict):
                raise ValueError(f"Unexpected nested structure in {path}: {type(ep[0])}")
            steps.extend(ep)
        if not steps:
            raise ValueError(f"{path} contained only empty episodes.")
    else:
        raise ValueError(f"Unexpected element type in {path}: {type(data[0])}")

    # Basic validation
    for i, s in enumerate(steps[:5]):
        if "image_primary" not in s or "action" not in s or "language_instruction" not in s:
            raise ValueError(f"Step {i} missing keys. Got keys: {list(s.keys())}")
        a = np.asarray(s["action"])
        if a.shape[-1] != 7:
            raise ValueError(f"Step {i} action must be 7D. Got shape {a.shape}")

    return steps


def compute_dataset_statistics_from_samples(
    samples: List[Dict[str, Any]],
    *,
    proprio_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Replicates OpenVLA TF stats schema:
      metadata = {
        "action": {"mean","std","max","min","q01","q99"},
        "proprio": {...},
        "num_transitions": N,
        "num_trajectories": num_trajectories
      }

    Here, we treat each sample as a transition; num_trajectories can be 1 for overfit,
    or you can pass pre-grouped episodes and compute it separately.
    """
    if not samples:
        raise ValueError("No samples provided for statistics.")

    # actions: [N, action_dim]
    A = np.stack([np.asarray(s["action"], dtype=np.float32).reshape(-1) for s in samples], axis=0)

    action_dim = A.shape[1]
    if proprio_dim is None:
        # mimic tf.zeros_like(traj["action"]) if no proprio
        proprio_dim = action_dim

    # proprio: [N, proprio_dim]
    P_list = []
    for s in samples:
        obs = s.get("observation", {})
        if "proprio" in obs:
            p = np.asarray(obs["proprio"], dtype=np.float32).reshape(proprio_dim)
        else:
            p = np.zeros((proprio_dim,), dtype=np.float32)
        P_list.append(p)
    P = np.stack(P_list, axis=0)

    def stats(x: np.ndarray) -> Dict[str, Any]:
        return {
            "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(),
            "max": x.max(0).tolist(),
            "min": x.min(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }

    return {
        "action": stats(A),
        "proprio": stats(P),
        "num_transitions": int(A.shape[0]),
        "num_trajectories": 1,  # for overfit; update if you have multiple episodes
    }


class TorchRLDSLikeIterableDataset(IterableDataset):
    """
    Yields dicts shaped exactly like an RLDS 'frame batch' expected by RLDSBatchTransform:
      dataset_name: str
      action: float32 array shaped (1, action_dim)
      observation.image_primary: uint8 array shaped (1, H, W, 3)
      task.language_instruction: bytes
    """

    def __init__(
        self,
        *,
        samples: List[Dict[str, Any]],
        batch_transform,  # RLDSBatchTransform instance
        dataset_name: str,
        shuffle_buffer_size: int = 10_000,
        infinite: bool = True,
        seed: int = 0,
    ):
        """
        samples entries should be:
          {
            "image_primary": np.uint8[H,W,3],
            "action": np.float32[action_dim]  (e.g., 7),
            "language_instruction": str,
            # optional:
            # "proprio": np.float32[...]
          }
        """
        if not samples:
            raise ValueError("samples must be non-empty")

        self.samples = samples
        self.batch_transform = batch_transform
        self.dataset_name = dataset_name
        self.shuffle_buffer_size = shuffle_buffer_size
        self.infinite = infinite
        self.seed = seed

        # For OpenVLA finetune script compatibility:
        self.dataset_statistics = compute_dataset_statistics_from_samples(
            [
                {
                    "action": s["action"],
                    "observation": {"proprio": s.get("proprio")} if "proprio" in s else {},
                }
                for s in samples
            ]
        )
        self.dataset_length = len(samples)

        # If you later have multiple trajectories, update num_trajectories accordingly:
        # self.dataset_statistics["num_trajectories"] = <num_episodes>

    def __len__(self):
        return self.dataset_length

    def _to_rlds_batch(self, s: Dict[str, Any]) -> Dict[str, Any]:
        img = np.asarray(s["image_primary"], dtype=np.uint8)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"image_primary must be HxWx3 uint8, got shape={img.shape}, dtype={img.dtype}")

        action = np.asarray(s["action"], dtype=np.float32).reshape(-1)
        lang = s["language_instruction"]
        if isinstance(lang, str):
            lang_b = lang.encode("utf-8")
        elif isinstance(lang, (bytes, bytearray)):
            lang_b = bytes(lang)
        else:
            raise TypeError(f"language_instruction must be str/bytes, got {type(lang)}")

        # Add leading batch dimension because transform indexes [0]
        rlds_batch = {
            "dataset_name": self.dataset_name,
            "action": action[None, :],  # (1, action_dim)
            "observation": {
                "image_primary": img[None, ...],  # (1, H, W, 3)
            },
            "task": {
                "language_instruction": lang_b,
            },
        }
        return rlds_batch

    def __iter__(self):
        rng = random.Random(self.seed)
        buf: List[Dict[str, Any]] = []

        while True:
            for s in self.samples:
                buf.append(s)
                if len(buf) >= self.shuffle_buffer_size:
                    idx = rng.randrange(len(buf))
                    out = buf.pop(idx)
                    yield self.batch_transform(self._to_rlds_batch(out))

            # drain buffer at end of pass
            while buf:
                idx = rng.randrange(len(buf))
                out = buf.pop(idx)
                yield self.batch_transform(self._to_rlds_batch(out))

            if not self.infinite:
                break
