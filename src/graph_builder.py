import torch
import librosa
import numpy as np
from torch_geometric.data import Data
from audio_features import extract_fast_segment_features

def build_graph_fast(path: str, label_vec: np.ndarray, clip_id: int, sr: int = 22050, clip_duration: float = 29.0, sim_threshold: float = 0.85, max_sim_edges: int = 3) -> Data:
    """Builds PyG Data graph using temporal + cosine similarity edges."""
    try:
        y, sr = librosa.load(path, sr=sr, duration=clip_duration)
    except Exception:
        return None

    if len(y) < int(sr * 3.0 * 2):
        return None

    feats = extract_fast_segment_features(y, sr=sr)
    if feats is None:
        return None

    n = feats.shape[0]
    norm_feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim = norm_feats @ norm_feats.T

    src, dst = [], []
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]

    for i in range(n):
        sims = sorted(
            [(j, sim[i, j]) for j in range(n) if abs(j - i) > 1],
            key=lambda t: -t[1]
        )
        for j, s in sims[:max_sim_edges]:
            if s >= sim_threshold:
                src.append(i)
                dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(feats, dtype=torch.float)
    y_label = torch.tensor(label_vec, dtype=torch.float).unsqueeze(0)

    data = Data(x=x, edge_index=edge_index, y=y_label)
    data.clip_id = clip_id
    data.num_nodes = n
    return data
