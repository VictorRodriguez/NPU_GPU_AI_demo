"""Explicit online preparation; runtime inference never downloads models."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from core import ROOT, read_config, validate_model

MODELS = {
    "asr_model": "OpenVINO/whisper-small-fp16-ov",
    "llm_model": "OpenVINO/Qwen3-4B-int4-ov",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Resolve new revisions instead of reusing the saved manifest")
    args = parser.parse_args()
    config = read_config()
    manifest_path = ROOT / "models/manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = dict(old)
    api = HfApi()
    for key, repo in MODELS.items():
        previous = old.get(key, {})
        revision = previous.get("revision") if previous.get("repo") == repo and not args.refresh else None
        revision = revision or api.model_info(repo).sha
        print(f"Downloading {repo} at {revision}. This may take several minutes.", flush=True)
        destination = ROOT / config[key]
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination),
                          ignore_patterns=["*.onnx", "*.pt", "*.safetensors", "*.msgpack", "*.h5"])
        validate_model(destination)
        manifest[key] = {"repo": repo, "revision": revision, "path": config[key],
                         "downloaded_at": datetime.now(timezone.utc).isoformat()}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Ready. Next: python preflight.py --warmup")

if __name__ == "__main__":
    main()
