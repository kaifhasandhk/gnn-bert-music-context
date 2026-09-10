import librosa
import numpy as np

def segment_audio(y: np.ndarray, sr: int = 22050, num_segments: int = 9):
    """Splits raw audio signal into N temporal non-overlapping segments."""
    segment_len = len(y) // num_segments
    segments = []
    for i in range(num_segments):
        chunk = y[i * segment_len : (i + 1) * segment_len]
        segments.append(chunk)
    return segments

def extract_segment_features(chunk: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extracts 152-dim node feature vector (16 STFT + 128 Mel + 8 Chroma)."""
    # STFT Statistics (16 dim)
    stft = np.abs(librosa.stft(chunk))
    stft_mean = np.mean(stft[:8], axis=1)
    stft_std = np.std(stft[:8], axis=1)
    stft_feat = np.concatenate([stft_mean, stft_std])

    # Mel-Spectrogram (128 dim)
    mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.mean(mel_db, axis=1)

    # Chroma Pitch Features (8 dim)
    chroma = librosa.feature.chroma_stft(y=chunk, sr=sr, n_chroma=8)
    chroma_feat = np.mean(chroma, axis=1)

    # Concatenate into 152-dimensional array
    return np.concatenate([stft_feat, mel_feat, chroma_feat])
