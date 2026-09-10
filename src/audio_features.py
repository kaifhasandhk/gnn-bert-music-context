import numpy as np
import librosa

def extract_fast_segment_features(y: np.ndarray, sr: int = 22050, seg_duration: float = 3.0, n_mels: int = 40, n_chroma: int = 12):
    """Vectorized single-pass extraction producing a 104-dim node feature matrix."""
    hop_length = 512
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)

    frames_per_seg = int((seg_duration * sr) / hop_length)
    total_frames = mel_db.shape[1]
    n_segments = total_frames // frames_per_seg

    if n_segments < 2:
        return None

    feats = []
    for i in range(n_segments):
        start_f = i * frames_per_seg
        end_f = (i + 1) * frames_per_seg

        m_seg = mel_db[:, start_f:end_f]
        c_seg = chroma[:, start_f:end_f]

        feat = np.concatenate([
            m_seg.mean(axis=1), m_seg.std(axis=1),
            c_seg.mean(axis=1), c_seg.std(axis=1)
        ])
        feats.append(feat)

    return np.stack(feats)
