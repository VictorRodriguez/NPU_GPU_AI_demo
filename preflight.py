"""Check local models/devices and optionally exercise both inference pipelines."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from core import Engine, ROOT, read_config, validate_device, validate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", action="store_true", help="Compile both models and generate a short answer")
    parser.add_argument("--audio", type=Path, help="Also transcribe an actual WAV; recommended for a full rehearsal")
    args = parser.parse_args()
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": sys.version, "status": "failed"}
    try:
        cfg = read_config()
        report["config"] = cfg
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ["openvino", "openvino-genai", "streamlit", "numpy", "soundfile"]}
        engine = Engine(cfg)
        report["devices"] = engine.available
        report["device_names"] = {d: engine.core.get_property(d, "FULL_DEVICE_NAME") for d in engine.available}
        for kind in ["asr", "llm"]:
            validate_device(cfg[kind + "_device"], engine.available)
            validate_model(ROOT / cfg[kind + "_model"])
        report["check"] = "Device enumeration and model files only; inference not yet tested."
        if args.warmup:
            print("Loading NPU/GPU models. First compilation can take several minutes.", flush=True)
            engine.load_asr()
            engine.load_llm()
            _, stats = engine.generate("El cliente necesita 50 laptops para vendedores que viajan.")
            report["generation"] = stats
            report["check"] = "Both pipelines loaded; LLM inference passed. ASR inference not yet tested."
        if args.audio:
            transcript, stats = engine.transcribe(args.audio.read_bytes())
            report["transcription"] = stats
            print("Transcript:", transcript)
            _, stats = engine.generate(transcript)
            report["generation"] = stats
            report["check"] = "Full WAV transcription and follow-up generation passed."
        report["status"] = "passed"
        print(report["check"])
    except Exception as exc:
        report["error"] = str(exc)
        print(f"FAILED: {exc}", file=sys.stderr)
    finally:
        path = ROOT / "preflight-report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report: {path}")
    return 0 if report["status"] == "passed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
