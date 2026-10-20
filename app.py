"""Run with: python -m streamlit run app.py"""
import os
# Model downloads are a separate setup step, never part of a live meeting.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
import hashlib
import streamlit as st
from core import Engine, ROOT, read_config

st.set_page_config(page_title="De la reunión a la acción", page_icon="💻", layout="wide")
st.markdown("""<style>
.block-container {max-width:1400px;padding-top:2rem;}
h1 {font-size:3rem!important;} textarea {font-size:1.15rem!important;}
.stButton>button {min-height:3.2rem;font-size:1.1rem;}
</style>""", unsafe_allow_html=True)
st.title("De la reunión a la acción")
st.subheader("Un asistente local para el trabajo comercial")
st.caption("Demo en un equipo Dell con tecnología Intel · Contenido ficticio · Borradores para revisión")

@st.cache_resource
def engine():
    return Engine(read_config())

try:
    runtime = engine()
except Exception as exc:
    st.error(f"No se pudo iniciar OpenVINO: {exc}")
    st.stop()

with st.sidebar:
    st.header("Preparación")
    st.write("Dispositivos detectados:", ", ".join(runtime.available) or "Ninguno")
    st.caption("Los indicadores identifican el dispositivo solicitado explícitamente; no son medidores de utilización.")
    st.write(f"Voz: **{runtime.config['asr_device']}** · Asistente: **{runtime.config['llm_device']}**")
    if st.button("Cargar modelos antes de presentar", use_container_width=True):
        try:
            with st.spinner("Cargando modelos; la primera compilación puede tardar varios minutos…"):
                runtime.load_asr()
                runtime.load_llm()
            st.success("Modelos cargados. Ejecute también el flujo completo antes de salir al escenario.")
        except Exception as exc:
            st.error(str(exc))
    st.divider()
    st.write("Los modelos se cargan desde disco. El audio y el texto se procesan en este equipo; no hay llamadas a una API de IA.")
    st.caption("Esto no verifica ni controla otros servicios de Windows o las políticas corporativas.")

for key, value in {"transcript":"", "answer":"", "asr_stats":None, "llm_stats":None,
                   "answer_inputs":None, "source_hash":None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

left, right = st.columns([1, 1.1], gap="large")
with left:
    st.markdown("### 1 · Escuchemos al cliente")
    source = st.radio("Fuente", ["Archivo WAV", "Micrófono"], horizontal=True)
    if source == "Archivo WAV":
        recording = st.file_uploader("Reunión grabada · máximo 120 segundos", type=["wav"])
    else:
        recording = st.audio_input("Grabe una conversación breve")
    blob = recording.getvalue() if recording else None
    fingerprint = hashlib.sha256(blob).hexdigest() if blob else None
    if fingerprint != st.session_state.source_hash:
        st.session_state.source_hash = fingerprint
        st.session_state.asr_stats = None
        # A newly selected source cannot inherit a prior source's transcript/result.
        st.session_state.transcript = ""
        st.session_state.answer = ""
        st.session_state.llm_stats = None
        st.session_state.answer_inputs = None
    if blob:
        st.audio(blob, format="audio/wav")
    if st.button("Transcribir reunión", type="primary", disabled=not blob, use_container_width=True):
        try:
            with st.spinner("Transcribiendo en el dispositivo seleccionado…"):
                text, stats = runtime.transcribe(blob)
            st.session_state.transcript = text
            st.session_state.asr_stats = stats
            st.session_state.answer = ""
            st.session_state.answer_inputs = None
            st.session_state.llm_stats = None
        except Exception as exc:
            st.error(f"No se completó la transcripción: {exc}")
    if st.button("Usar guion ficticio como texto de respaldo", use_container_width=True):
        st.session_state.transcript = (ROOT / "assets/customer_script.txt").read_text(encoding="utf-8")
        st.session_state.asr_stats = None
        st.session_state.answer = ""
        st.session_state.answer_inputs = None
        st.session_state.llm_stats = None
    transcript = st.text_area("Transcripción editable", key="transcript", height=230, max_chars=10000)
    stats = st.session_state.asr_stats
    if stats:
        st.success(f"Voz → {stats['device']} · {stats['seconds']:.1f} s de inferencia · "
                   f"audio de {stats['audio_seconds']:.1f} s · carga {stats['load_seconds']:.1f} s")
        st.caption("Si edita el texto, el tiempo anterior corresponde a la transcripción original.")
    else:
        st.caption("Sin transcripción medida. El texto de respaldo no demuestra ejecución en NPU.")

with right:
    st.markdown("### 2 · Preparemos el seguimiento")
    adjustment = st.text_input("Cambio solicitado por el cliente", placeholder="Ahora necesitamos 100 laptops", max_chars=1000)
    if st.button("Generar seguimiento", type="primary", disabled=not transcript.strip(), use_container_width=True):
        st.session_state.answer = ""
        st.session_state.answer_inputs = None
        st.session_state.llm_stats = None
        try:
            with st.spinner("Preparando requisitos, próximos pasos y correo…"):
                answer, stats = runtime.generate(transcript, adjustment)
            st.session_state.answer = answer
            st.session_state.llm_stats = stats
            st.session_state.answer_inputs = (transcript, adjustment)
        except Exception as exc:
            st.error(f"No se completó la generación: {exc}")
    if st.session_state.answer:
        stale = st.session_state.answer_inputs != (transcript, adjustment)
        if stale:
            st.warning("Los datos cambiaron. Genere de nuevo para actualizar este borrador.")
        stats = st.session_state.llm_stats
        st.success(f"Asistente → {stats['device']} · {stats['seconds']:.1f} s de inferencia · carga {stats['load_seconds']:.1f} s")
        st.markdown(st.session_state.answer)
        st.download_button("Descargar borrador (.md)", st.session_state.answer,
                           file_name="seguimiento_cliente.md", mime="text/markdown", disabled=stale)
    else:
        st.info("Aquí aparecerán las necesidades, preguntas pendientes, próximos pasos y un correo de seguimiento.")
