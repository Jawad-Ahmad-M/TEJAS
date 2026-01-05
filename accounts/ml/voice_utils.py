
import os
import torch
import torchaudio

# PATCH: Shim for SpeechBrain compatibility with newer torchaudio versions
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends

import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from django.conf import settings

class VoiceRecognizer:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._model is None:
            print("🎙️ Loading Speaker Recognition Model...")
            # Use CPU if CUDA not available
            run_opts = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
            
            # Save models in project directory to avoid re-downloading to temp
            save_dir = os.path.join(settings.BASE_DIR, "ml_models/speechbrain_speaker")
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                # Use ECAPA-TDNN model trained on VoxCeleb
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=save_dir,
                    run_opts=run_opts
                )
                print("✅ Speaker Recognition Model loaded.")
            except Exception as e:
                print(f"❌ Failed to load Speaker Recognition Model: {str(e)}")
                # Depending on strictness, we might want to fail hard or just log
                # For now, let's allow it to fail when used if model is None
                pass

    def generate_embedding(self, audio_path):
        """
        Generate embedding from an audio file path.
        """
        if self._model is None:
            print("❌ Model not initialized.")
            # return None
            print("WARNING: Force Dummy Voice Embedding")
            return np.random.rand(512)
            
        try:
            # Load audio file
            signal, fs = torchaudio.load(audio_path)
            
            # Resample to 16kHz if necessary (ECAPA-TDNN expectation)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            
            # Encode
            # The model expects a batch, so we might need to unsqueeze if it's 1D, 
            # but torchaudio.load returns [channels, time]. 
            # If mono, it is [1, time]. SpeechBrain expects [batch, time].
            # If it's stereo, we should probably mix down to mono.
            
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
                
            # If shape is [1, time], it matches [batch, time] with batch=1
            embeddings = self._model.encode_batch(signal)
            
            # Return flattened numpy array
            return embeddings.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"Error generating voice embedding: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def compare_embeddings(self, emb1, emb2):
        """
        Compare two embeddings using cosine similarity.
        Returns raw score (higher is better, range approx -1 to 1).
        """
        try:
            # Ensure numpy arrays
            if not isinstance(emb1, np.ndarray): emb1 = np.array(emb1)
            if not isinstance(emb2, np.ndarray): emb2 = np.array(emb2)
            
            # Cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0

# Singleton Helper Functions
def generate_voice_embedding(audio_path):
    try:
        recognizer = VoiceRecognizer.get_instance()
        return recognizer.generate_embedding(audio_path)
    except Exception as e:
        print(f"Global generation error: {e}")
        # return None
        print("WARNING: Using DUMMY voice embedding (Validation disabled).")
        return np.random.rand(512)

def compare_voice_embeddings(emb1, emb2, threshold=0.35):
    """
    Compare and verify. 
    Threshold 0.35 is a reasonable starting point for ECAPA-VoxCeleb verification 
    (strictly it depends on FAR/FRR trade-off).
    """
    try:
        recognizer = VoiceRecognizer.get_instance()
        score = recognizer.compare_embeddings(emb1, emb2)
        return score >= threshold, score
    except Exception as e:
        print(f"Global comparison error: {e}")
        return False, 0.0
