# AI PC DEMO

A Windows-native local Python demo for a commercial presentation. Upload a fictional customer meeting WAV, transcribe it on the Intel NPU, and generate Spanish requirements, questions, next steps, and a follow-up email on the integrated Intel GPU. A presenter can change a requirement and regenerate.

This is a working project implementation, **not yet validated on your specific laptop**. CPU-side code, audio handling, state transitions, and fail-closed device routing have automated checks. Final NPU/GPU support, model compilation, memory use, transcription accuracy, and stage latency require the hardware rehearsal below. No performance or battery-life claims are built in.

## 1. Windows prerequisites

- Windows 11 x64 on your Panther Lake Dell laptop; 32 GB system RAM.
- Python **3.11 x64** from https://www.python.org/downloads/windows/ (include the Python Launcher).
- VS Code with Microsoft's Python extension.
- Current Dell-approved graphics and Intel AI Boost/NPU drivers for the exact Dell model. Check Device Manager for Intel graphics and Intel AI Boost. Reboot after driver changes.
- Internet for pip installation and the initial model downloads. Allow several GB of model downloads, disk space for dependencies and caches, and time for initial compilation.

Run everything below in **Windows PowerShell, not the WSL terminal**. In VS Code use Terminal → Select Default Profile → PowerShell. Extract the ZIP to a writable local folder, then File → Open Folder → `dell_ai_demo` (the folder containing this README).

## 2. Install

Open the VS Code PowerShell terminal in that folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Select `.venv\Scripts\python.exe` using Ctrl+Shift+P → Python: Select Interpreter. Activation is optional: the explicit interpreter paths above avoid PowerShell activation-policy problems. Alternatively run `setup.ps1` if your organization's policy allows local scripts; the commands above require no policy changes.

The version ranges request stable GenAI 2026.1 or later; pip resolves its compatible OpenVINO/tokenizer dependencies. After a successful rehearsal, save the exact environment and do not update it before presenting:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-installed.txt
```

If pip cannot find a compatible release, confirm Python is 3.11 **64-bit** and your corporate package mirror contains the required versions. Do not switch to nightly builds during event preparation without retesting.

## 3. Download the two models (once, online)

```powershell
.\.venv\Scripts\python.exe download_models.py
```

- Speech: `OpenVINO/whisper-small-fp16-ov` (multilingual; default language Spanish).
- Assistant: `OpenVINO/Qwen3-4B-int4-ov` (compact local multilingual model).

No OpenAI API key, cloud inference account, PyTorch installation, or paid inference API is required. Hugging Face hosts the initial downloads. Network/proxy policy still applies. Downloaded model revisions are recorded in `models/manifest.json` and reused on retries. `--refresh` explicitly selects newer revisions. Model weights and dependencies are **not included in this ZIP**. Models retain their upstream licenses; see their cards linked below.

## 4. Check your devices

```powershell
.\.venv\Scripts\python.exe preflight.py
.\.venv\Scripts\python.exe preflight.py --warmup
```

The first command checks files and device enumeration. `--warmup` loads/compiles both pipelines and exercises the LLM; it does **not** claim to test speech inference. Compilation may take minutes. The JSON report distinguishes these checks. Nothing automatically falls back to CPU. If `GPU` or `NPU` is missing, fix Windows drivers before the demo.

For complete validation, use your actual audio:

```powershell
.\.venv\Scripts\python.exe preflight.py --audio assets\customer_meeting.wav
```

This transcribes the WAV and generates a follow-up. It prints the transcript locally and writes timing/device metadata to `preflight-report.json`; it does not put the conversation in that report.

## 5. Prepare the recording

Read `assets/customer_script.txt` aloud, or have someone read it. Record approximately 45–60 seconds. The browser microphone option can capture a WAV directly, or upload a PCM WAV from another recording tool. Stereo/mono and common sample rates are converted to 16 kHz mono automatically. Keep audio under 120 seconds and 20 MB.

MP3, M4A, MP4 and video are not accepted directly. If your recorder produces those formats, export/convert to WAV before presenting. With an existing FFmpeg installation:

```powershell
ffmpeg -i meeting.m4a -ar 16000 -ac 1 -c:a pcm_s16le assets\customer_meeting.wav
```

No recording or synthetic audio is supplied in this ZIP. The bundled fictional script can be loaded as an explicitly labeled text backup; that path skips speech recognition and makes no NPU execution claim.

## 6. Start the interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or double-click `run_demo.bat`. Open **http://127.0.0.1:8501** in Edge. Microphone recording may request browser and Windows microphone permission; WAV upload avoids that dependency. Keep the terminal running. Ctrl+C stops the app.

1. Use the sidebar to load models before presenting.
2. Upload your customer WAV, play it, and click **Transcribir reunión**.
3. Review/correct the Spanish transcript, then click **Generar seguimiento**.
4. Enter **Ahora necesitamos 100 laptops en lugar de 50** and generate again.
5. Optionally download the resulting Markdown draft. No email is sent.

The app is bound to localhost, Streamlit usage statistics are disabled, and model inference uses only local paths with Hugging Face offline mode enabled. Uploaded audio, transcripts, and drafts are kept in process/browser-session memory rather than automatically written to disk; explicit downloads are saved by your browser. Restarting the app process clears backend session memory. This app does not control Windows telemetry, corporate monitoring, browser extensions, or other applications. Demonstrate offline operation by disconnecting network access after preparation and running the entire flow again.

## 7. Presenter rehearsal

Use AC power and the same Windows power mode, projector resolution, and browser zoom as the event. Close unnecessary applications. Run the full workflow once in the **same running Streamlit process** before the talk; CLI warmup cannot keep the app's models resident. Device compilation/loading time is shown separately from pipeline inference time. These are application timings, not pure hardware kernel measurements, and there is no claimed speedup without a controlled comparison.

GPU/NPU labels identify explicitly selected pipeline targets; they are **not measured utilization**. The CPU still runs the interface and orchestration. The speech and text steps are sequential, so this workflow does **not** claim simultaneous GPU/NPU inference. For optional live activity, use Windows Task Manager → Performance → NPU/GPU; a short workload may finish too quickly to appear clearly in the sampling interval.

Keep a screen recording of a successful rehearsal as backup. Disable notifications and avoid showing real customer data. The app's fictional-text backup is clearly labeled and should be described honestly if used.

## Configuration and troubleshooting

Edit `config.json` only between runs and restart Streamlit afterward.

- Default targets are `NPU` for speech and `GPU` for the LLM. For multiple GPUs, choose the device ID shown by preflight (for example `GPU.0`); verify its full name in the report.
- For diagnostic CPU runs, set a target to `CPU` explicitly. The interface/report will say CPU. Restore NPU/GPU and rehearse before presenting.
- Language is `es`; use `en` for English audio. This changes ASR language; the assistant prompt remains Spanish.
- Missing/incomplete model: rerun `download_models.py` online. Downloads resume at the saved revision.
- NPU compilation error: verify Dell/Intel driver and OpenVINO compatibility; copy the full error for troubleshooting. A detected device alone does not prove every model compiles.
- GPU memory issue: close other memory-heavy apps. Shared system RAM is not all available to the GPU. Do not assume 32 GB guarantees any model will fit.
- Slow first run: allow compilation to finish, then rehearse again. No speed or accuracy target is promised.
- Browser changes or inputs clear the prior source's transcript; changed text marks old output stale and disables its export until regenerated.
- Generated output needs review. The prompt discourages fabricated pricing/product details, but this is not a validated Dell product recommendation engine or a pricing tool.

## Project files

`app.py`: Spanish browser UI. `core.py`: audio, prompts and inference. `download_models.py`: explicit preparation. `preflight.py`: hardware rehearsal/report. `config.json`: local paths/device targets. `tests/`: host-side regression checks. `.vscode/`: interpreter preference. `.streamlit/`: local server/theme configuration.

Run host-side tests after installation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Primary references

- https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
- https://github.com/openvinotoolkit/openvino.genai
- https://huggingface.co/OpenVINO/whisper-small-fp16-ov
- https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov

This is a custom demonstration, not an official Dell application or an endorsement by Dell.
