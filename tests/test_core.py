import io
import unittest
from unittest.mock import patch
import numpy as np
from core import clean_answer, decode_wav, make_prompt, validate_device, Engine

class CoreTests(unittest.TestCase):
    def test_device_routing_fails_closed(self):
        validate_device("GPU", ["CPU", "GPU.0", "NPU"])
        with self.assertRaises(RuntimeError):
            validate_device("NPU", ["CPU", "GPU.0"])
        with self.assertRaises(RuntimeError):
            validate_device("GPU.1", ["CPU", "GPU.0"])

    def test_prompt_requires_input_and_does_not_drop_changed_requirement(self):
        with self.assertRaises(ValueError):
            make_prompt("  ")
        prompt = make_prompt("Necesitamos 50 laptops", "Ahora 100 laptops")
        self.assertIn("Ahora 100 laptops", prompt)
        self.assertIn("No inventes precios", prompt)
        with self.assertRaises(ValueError):
            make_prompt("x" * 10001)

    def test_thinking_never_becomes_customer_draft(self):
        self.assertEqual(clean_answer("<think>internal</think>Correo listo"), "Correo listo")
        with self.assertRaises(RuntimeError):
            clean_answer("<think>unfinished")

    def test_stereo_48khz_becomes_16khz_mono(self):
        import soundfile as sf
        buf = io.BytesIO()
        t = np.arange(48000) / 48000
        samples = np.stack([0.1 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
        sf.write(buf, samples, 48000, format="WAV")
        audio, duration = decode_wav(buf.getvalue())
        self.assertEqual(audio.shape, (16000,))
        self.assertAlmostEqual(duration, 1)
        self.assertEqual(audio.dtype, np.float32)

    def test_silence_and_long_audio_rejected(self):
        import soundfile as sf
        for seconds in [1, 121]:
            buf = io.BytesIO()
            sf.write(buf, np.zeros(8000 * seconds), 8000, format="WAV")
            with self.assertRaises(ValueError):
                decode_wav(buf.getvalue())

    def test_missing_npu_prevents_pipeline_construction(self):
        import sys
        import types
        fake = types.SimpleNamespace(WhisperPipeline=unittest.mock.Mock())
        with patch.dict(sys.modules, {"openvino_genai": fake}):
            # Host-side device guard can be tested without GPU/NPU access.
            engine = Engine.__new__(Engine)
            engine.config = {"asr_device": "NPU"}
            engine.available = ["CPU"]
            engine.asr = None
            import threading
            engine.lock = threading.RLock()
            with self.assertRaises(RuntimeError):
                engine.load_asr()
            fake.WhisperPipeline.assert_not_called()

if __name__ == "__main__":
    unittest.main()
