import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from core import Engine, ROOT

class AppTests(unittest.TestCase):
    def test_text_backup_generate_and_stale_output(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        backup = next(b for b in app.button if "guion ficticio" in b.label)
        backup.click().run()
        self.assertIn("cincuenta", app.text_area[0].value)
        self.assertIsNone(app.session_state['asr_stats'])
        with patch.object(Engine, "generate", return_value=("## Necesidades del cliente\n50 equipos", {
            "device":"GPU", "seconds":1.2, "load_seconds":0.1})):
            next(b for b in app.button if b.label == "Generar seguimiento").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state['answer'])
        app.text_input[0].set_value("Ahora 100 laptops").run()
        self.assertTrue(any("Los datos cambiaron" in w.value for w in app.warning))

    def test_missing_hardware_error_is_visible(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        with patch.object(Engine, "load_asr", side_effect=RuntimeError("NPU no disponible")):
            next(b for b in app.button if "Cargar modelos" in b.label).click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("NPU no disponible" in e.value for e in app.error))

if __name__ == '__main__':
    unittest.main()
