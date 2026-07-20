import streamlit as st
import time
import os
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ── Detect environment ────────────────────────────────────────────────────────
IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") == "1" or \
           os.path.exists("/mount/src")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetingMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0D1117; color: #E6EDF3; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }

[data-testid="stSidebar"] { background: #161B22; border-right: 1px solid #21262D; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.app-logo { display:flex;align-items:center;gap:10px;padding:0.5rem 0 1.5rem;border-bottom:1px solid #21262D;margin-bottom:1.5rem; }
.app-logo-text { font-size:18px;font-weight:700;color:#E6EDF3;letter-spacing:-0.3px; }
.app-logo-sub { font-size:11px;color:#7D8590;font-weight:400;letter-spacing:0.3px;text-transform:uppercase; }
.nav-section { font-size:10px;font-weight:600;color:#7D8590;text-transform:uppercase;letter-spacing:0.8px;padding:0.5rem 0 0.4rem;margin-bottom:0.2rem; }

.info-card { background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1rem;transition:border-color 0.2s; }
.info-card:hover { border-color:#388BFD; }
.card-header { display:flex;align-items:center;gap:8px;margin-bottom:0.75rem; }
.card-icon { font-size:16px; }
.card-title { font-size:13px;font-weight:600;color:#7D8590;text-transform:uppercase;letter-spacing:0.5px; }
.card-content { font-size:14px;line-height:1.7;color:#C9D1D9; }

.hero-section { background:linear-gradient(135deg,#161B22 0%,#0D1117 100%);border:1px solid #21262D;border-radius:14px;padding:2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden; }
.hero-section::before { content:'';position:absolute;top:-60px;right:-60px;width:200px;height:200px;background:radial-gradient(circle,rgba(56,139,253,0.15) 0%,transparent 70%);border-radius:50%; }
.hero-title { font-size:28px;font-weight:700;color:#E6EDF3;letter-spacing:-0.5px;margin-bottom:6px; }
.hero-sub { font-size:15px;color:#7D8590;font-weight:400;margin-bottom:1.5rem; }
.hero-badge { display:inline-flex;align-items:center;gap:5px;background:rgba(56,139,253,0.15);border:1px solid rgba(56,139,253,0.3);border-radius:20px;padding:4px 12px;font-size:12px;color:#388BFD;font-weight:500;margin-right:6px;margin-bottom:6px; }

.warn-box { background:#1C1A00;border:1px solid #D29922;border-left:3px solid #D29922;border-radius:8px;padding:10px 14px;font-size:13px;color:#E3B341;margin-bottom:14px; }
.success-box { background:#1C2B1C;border:1px solid #3FB950;border-radius:8px;padding:10px 14px;font-size:13px;color:#3FB950;margin-top:8px; }

.step-row { display:flex;align-items:center;gap:12px;padding:10px 14px;background:#0D1117;border-radius:8px;margin-bottom:6px;border:1px solid #21262D; }
.step-icon { font-size:16px;width:24px;text-align:center; }
.step-label { font-size:13px;color:#C9D1D9;flex:1; }
.step-done { color:#3FB950; }
.step-active { color:#388BFD; }
.step-wait { color:#484F58; }

.action-item { display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:#0D1117;border:1px solid #21262D;border-left:3px solid #3FB950;border-radius:0 8px 8px 0;margin-bottom:8px;font-size:14px;color:#C9D1D9;line-height:1.6; }
.decision-item { display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:#0D1117;border:1px solid #21262D;border-left:3px solid #D29922;border-radius:0 8px 8px 0;margin-bottom:8px;font-size:14px;color:#C9D1D9;line-height:1.6; }
.question-item { display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:#0D1117;border:1px solid #21262D;border-left:3px solid #BC8CFF;border-radius:0 8px 8px 0;margin-bottom:8px;font-size:14px;color:#C9D1D9;line-height:1.6; }

.chat-wrap { background:#0D1117;border:1px solid #21262D;border-radius:12px;padding:1.25rem;margin-bottom:1rem;max-height:480px;overflow-y:auto; }
.msg-user { display:flex;justify-content:flex-end;margin-bottom:12px; }
.msg-user .bubble { background:#1F6FEB;color:#FFFFFF;border-radius:14px 14px 2px 14px;padding:10px 14px;max-width:75%;font-size:14px;line-height:1.6; }
.msg-bot { display:flex;justify-content:flex-start;gap:8px;margin-bottom:12px; }
.msg-bot .avatar { width:28px;height:28px;background:linear-gradient(135deg,#388BFD,#BC8CFF);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px; }
.msg-bot .bubble { background:#161B22;border:1px solid #21262D;color:#C9D1D9;border-radius:2px 14px 14px 14px;padding:10px 14px;max-width:75%;font-size:14px;line-height:1.6; }

.transcript-box { font-family:'JetBrains Mono',monospace;font-size:12.5px;line-height:1.9;color:#8B949E;background:#0D1117;border:1px solid #21262D;border-radius:10px;padding:1.25rem;max-height:350px;overflow-y:auto;white-space:pre-wrap;word-wrap:break-word; }

.stat-row { display:flex;gap:10px;flex-wrap:wrap;margin-bottom:1.25rem; }
.stat-pill { background:#161B22;border:1px solid #21262D;border-radius:8px;padding:10px 16px;flex:1;min-width:100px; }
.stat-pill .sv { font-size:22px;font-weight:700;color:#388BFD; }
.stat-pill .sl { font-size:11px;color:#7D8590;font-weight:500;text-transform:uppercase;letter-spacing:0.4px;margin-top:2px; }

.stTextInput>div>div>input, .stTextArea>div>div>textarea { background:#161B22!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;font-family:'Inter',sans-serif!important;font-size:14px!important; }
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus { border-color:#388BFD!important;box-shadow:0 0 0 3px rgba(56,139,253,0.1)!important; }
.stSelectbox>div>div { background:#161B22!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important; }

.stButton>button { background:#1F6FEB!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;font-weight:600!important;font-size:14px!important;padding:0.5rem 1.25rem!important;transition:background 0.2s,transform 0.1s!important; }
.stButton>button:hover { background:#388BFD!important;transform:translateY(-1px)!important; }
.secondary-btn>button { background:#21262D!important;color:#C9D1D9!important;border:1px solid #30363D!important; }
.secondary-btn>button:hover { background:#30363D!important; }

.stProgress>div>div>div>div { background:linear-gradient(90deg,#1F6FEB,#BC8CFF)!important;border-radius:99px!important; }
.stProgress>div>div { background:#21262D!important;border-radius:99px!important;height:6px!important; }

.stTabs [data-baseweb="tab-list"] { gap:4px;background:transparent;border-bottom:1px solid #21262D; }
.stTabs [data-baseweb="tab"] { background:transparent!important;color:#7D8590!important;border-radius:8px 8px 0 0!important;font-size:13px!important;font-weight:500!important;padding:8px 16px!important;border-bottom:2px solid transparent!important; }
.stTabs [aria-selected="true"] { color:#388BFD!important;border-bottom-color:#388BFD!important;background:rgba(56,139,253,0.08)!important; }

hr { border:none;border-top:1px solid #21262D;margin:1.25rem 0; }
::-webkit-scrollbar { width:5px;height:5px; }
::-webkit-scrollbar-track { background:#0D1117; }
::-webkit-scrollbar-thumb { background:#30363D;border-radius:99px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "home", "result": None, "processing": False,
        "chat_history": [], "source": "", "language": "english",
        "processing_log": [], "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="app-logo">
        <div style="font-size:28px">🧠</div>
        <div>
            <div class="app-logo-text">MeetingMind</div>
            <div class="app-logo-sub">AI Intelligence Layer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Environment badge
    if IS_CLOUD:
        st.markdown("""
        <div style="background:#1C1A00;border:1px solid #D29922;border-radius:6px;
                    padding:6px 10px;font-size:11px;color:#E3B341;margin-bottom:12px;">
            ☁️ Cloud Mode — Use file upload for best results
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1C2B1C;border:1px solid #3FB950;border-radius:6px;
                    padding:6px 10px;font-size:11px;color:#3FB950;margin-bottom:12px;">
            💻 Local Mode — YouTube URLs + file upload both work
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("🏠", "home", "Home"),
        ("📋", "results", "Analysis Results"),
        ("💬", "chat", "Ask Questions"),
        ("📄", "transcript", "Full Transcript"),
    ]

    for icon, key, label in pages:
        disabled = key in ("results", "chat", "transcript") and not st.session_state.result
        if disabled:
            st.markdown(
                f'<div style="padding:8px 10px;font-size:13px;color:#484F58;'
                f'cursor:not-allowed;">{icon} {label}</div>',
                unsafe_allow_html=True
            )
        else:
            if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    if st.session_state.result:
        st.markdown("---")
        st.markdown('<div class="nav-section">Session Stats</div>', unsafe_allow_html=True)
        transcript = st.session_state.result.get("transcript", "")
        word_count = len(transcript.split())
        action_count = len([l for l in st.session_state.result.get("action_items","").split("\n") if l.strip()])
        st.markdown(f"""
        <div style="font-size:12px;color:#7D8590;line-height:2.2;">
            📝 <b style="color:#C9D1D9;">{word_count:,}</b> words transcribed<br>
            ✅ <b style="color:#C9D1D9;">{action_count}</b> action items<br>
            💬 <b style="color:#C9D1D9;">{len(st.session_state.chat_history)}</b> questions asked
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("🔄  New Analysis", use_container_width=True):
            for k in ["result", "chat_history", "processing_log", "error"]:
                st.session_state[k] = [] if k in ("chat_history","processing_log") else None
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#484F58;text-align:center;">MeetingMind v3.0 · Powered by LLM</div>', unsafe_allow_html=True)


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline_ui(source: str, language: str):
    steps = [
        ("🎧", "Processing audio / video input"),
        ("🗣️", "Transcribing speech to text"),
        ("📌", "Generating title"),
        ("📋", "Summarizing content"),
        ("✅", "Extracting action items"),
        ("🔑", "Extracting key decisions"),
        ("❓", "Detecting open questions"),
        ("🔗", "Building RAG knowledge base"),
    ]

    status_ph = st.empty()
    progress_bar = st.progress(0)

    def update_ui(step_idx):
        with status_ph.container():
            for i, (icon, label) in enumerate(steps):
                if i < step_idx:
                    st.markdown(f'<div class="step-row"><span class="step-icon">{icon}</span><span class="step-label">{label}</span><span class="step-status step-done">✓ Done</span></div>', unsafe_allow_html=True)
                elif i == step_idx:
                    st.markdown(f'<div class="step-row"><span class="step-icon">{icon}</span><span class="step-label">{label}</span><span class="step-status step-active">⟳ Running…</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="step-row"><span class="step-icon" style="opacity:0.3">{icon}</span><span class="step-label" style="color:#484F58;">{label}</span><span class="step-status step-wait">—</span></div>', unsafe_allow_html=True)

    try:
        update_ui(0); progress_bar.progress(5)
        chunks = process_input(source)

        update_ui(1); progress_bar.progress(20)
        transcript = transcribe_all(chunks, language)

        update_ui(2); progress_bar.progress(35)
        title = generate_title(transcript)

        update_ui(3); progress_bar.progress(50)
        summary = summarize(transcript)

        update_ui(4); progress_bar.progress(62)
        action_items = extract_action_items(transcript)

        update_ui(5); progress_bar.progress(74)
        decisions = extract_key_decisions(transcript)

        update_ui(6); progress_bar.progress(86)
        questions = extract_questions(transcript)

        update_ui(7); progress_bar.progress(95)
        rag_chain = build_rag_chain(transcript)

        progress_bar.progress(100)
        status_ph.empty()

        return {
            "title": title, "transcript": transcript, "summary": summary,
            "action_items": action_items, "key_decisions": decisions,
            "open_questions": questions, "rag_chain": rag_chain,
        }

    except Exception as e:
        status_ph.empty()
        progress_bar.empty()
        raise e


# ── Page: Home ────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🧠 MeetingMind AI</div>
        <div class="hero-sub">Transform any video or audio into structured intelligence — instantly.</div>
        <span class="hero-badge">🎙️ Speech-to-Text</span>
        <span class="hero-badge">📋 Auto Summary</span>
        <span class="hero-badge">✅ Action Items</span>
        <span class="hero-badge">💬 Chat with Meeting</span>
        <span class="hero-badge">🔑 Key Decisions</span>
    </div>
    """, unsafe_allow_html=True)

    source = None

    # ── Two input methods as tabs ─────────────────────────────────────────
    tab_upload, tab_url = st.tabs(["📁 Upload File  (works everywhere)", "🔗 YouTube URL  (local only)"])

    with tab_upload:
        st.markdown('<div style="font-size:13px;color:#7D8590;margin-bottom:12px;">Upload any audio or video file — MP4, MP3, WAV, M4A, WEBM, OGG supported.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload file",
                type=["mp4", "mp3", "wav", "m4a", "webm", "ogg", "flac", "aac", "mkv"],
                label_visibility="collapsed",
            )
        with col2:
            lang_upload = st.selectbox(
                "Language",
                ["english", "hinglish", "hindi", "spanish", "french", "german"],
                label_visibility="collapsed",
                key="lang_upload",
            )

        if uploaded_file is not None:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = save_path
            language = lang_upload
            st.markdown(f'<div class="success-box">✅ Ready: <b>{uploaded_file.name}</b> ({round(uploaded_file.size/1024/1024,1)} MB)</div>', unsafe_allow_html=True)

    with tab_url:
        if IS_CLOUD:
            st.markdown('<div class="warn-box">⚠️ <b>Cloud deployment detected.</b> YouTube downloads are blocked by YouTube on cloud servers. Use the <b>Upload File</b> tab instead — download the video locally first, then upload it.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:13px;color:#7D8590;margin-bottom:12px;">Paste any YouTube URL — works when running locally.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            yt_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
        with col2:
            lang_url = st.selectbox(
                "Language",
                ["english", "hinglish", "hindi", "spanish", "french", "german"],
                label_visibility="collapsed",
                key="lang_url",
            )

        if yt_url.strip():
            source = yt_url.strip()
            language = lang_url

    # ── Analyse button ────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    col_btn, col_tip = st.columns([1, 3])
    with col_btn:
        analyze = st.button("⚡  Analyze", use_container_width=True)
    with col_tip:
        st.markdown('<div style="font-size:12px;color:#7D8590;padding-top:10px;">Upload tab works on all deployments. YouTube tab works locally only.</div>', unsafe_allow_html=True)

    if analyze:
        if not source:
            st.error("⚠️  Please upload a file or enter a YouTube URL.")
            return

        # Block YouTube on cloud
        if IS_CLOUD and source.startswith("http"):
            st.error(
                "❌ YouTube downloads are blocked on cloud deployments.\n\n"
                "**How to fix:** Download the video on your PC first, "
                "then use the **Upload File** tab to upload it here."
            )
            return

        st.session_state.source = source
        st.session_state.language = language
        st.session_state.chat_history = []
        st.session_state.error = None

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:600;color:#7D8590;margin-bottom:12px;letter-spacing:0.4px;text-transform:uppercase;">Processing Pipeline</div>', unsafe_allow_html=True)

        try:
            result = run_pipeline_ui(source, language)
            st.session_state.result = result
            st.success(f"✅ Analysis complete! Title: **{result['title']}**")
            time.sleep(0.8)
            st.session_state.page = "results"
            st.rerun()
        except Exception as e:
            err = str(e)
            if "403" in err or "blocked" in err.lower() or "youtube" in err.lower():
                st.error(
                    "❌ YouTube blocked this download.\n\n"
                    "**Fix:** Use the **Upload File** tab — "
                    "download the video to your PC first, then upload it."
                )
            else:
                st.error(f"❌ {err}")

    else:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:600;color:#7D8590;margin-bottom:14px;letter-spacing:0.4px;text-transform:uppercase;">What you get</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="info-card"><div class="card-header"><span class="card-icon">📋</span><span class="card-title">Smart Summary</span></div><div class="card-content">Concise structured summary of the entire meeting.</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="info-card"><div class="card-header"><span class="card-icon">✅</span><span class="card-title">Action Items</span></div><div class="card-content">Every task and follow-up automatically extracted.</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="info-card"><div class="card-header"><span class="card-icon">💬</span><span class="card-title">RAG Chat</span></div><div class="card-content">Ask anything — answers grounded in the transcript.</div></div>', unsafe_allow_html=True)


# ── Page: Results ─────────────────────────────────────────────────────────────
def page_results():
    result = st.session_state.result
    if not result:
        st.warning("No analysis yet. Go to Home and process a video first.")
        return

    title = result.get("title", "Untitled Meeting")
    transcript = result.get("transcript", "")
    summary = result.get("summary", "")
    action_items = result.get("action_items", "")
    decisions = result.get("key_decisions", "")
    questions = result.get("open_questions", "")

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.25rem;">
        <div style="font-size:22px;font-weight:700;color:#E6EDF3;">{title}</div>
        <div style="background:rgba(56,139,253,0.15);border:1px solid rgba(56,139,253,0.3);border-radius:20px;padding:3px 12px;font-size:12px;color:#388BFD;font-weight:500;">Analysis Complete</div>
    </div>
    """, unsafe_allow_html=True)

    word_count = len(transcript.split())
    action_count = len([l for l in action_items.split("\n") if l.strip()])
    decision_count = len([l for l in decisions.split("\n") if l.strip()])
    question_count = len([l for l in questions.split("\n") if l.strip()])

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill"><div class="sv">{word_count:,}</div><div class="sl">Words</div></div>
        <div class="stat-pill"><div class="sv">{action_count}</div><div class="sl">Actions</div></div>
        <div class="stat-pill"><div class="sv">{decision_count}</div><div class="sl">Decisions</div></div>
        <div class="stat-pill"><div class="sv">{question_count}</div><div class="sl">Questions</div></div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions"])

    with tabs[0]:
        st.markdown(f'<div class="info-card"><div class="card-content">{summary}</div></div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download Summary", summary, "summary.txt", "text/plain")

    with tabs[1]:
        lines = [l.strip() for l in action_items.split("\n") if l.strip()]
        for line in lines:
            st.markdown(f'<div class="action-item">☐  {line.lstrip("-•0123456789. ")}</div>', unsafe_allow_html=True)
        if lines:
            st.download_button("⬇️ Export Action Items", "\n".join(lines), "action_items.txt", "text/plain")

    with tabs[2]:
        lines = [l.strip() for l in decisions.split("\n") if l.strip()]
        for line in lines:
            st.markdown(f'<div class="decision-item">🔑  {line.lstrip("-•0123456789. ")}</div>', unsafe_allow_html=True)

    with tabs[3]:
        lines = [l.strip() for l in questions.split("\n") if l.strip()]
        for line in lines:
            st.markdown(f'<div class="question-item">❓  {line.lstrip("-•0123456789. ")}</div>', unsafe_allow_html=True)


# ── Page: Chat ────────────────────────────────────────────────────────────────
def page_chat():
    result = st.session_state.result
    if not result:
        st.warning("No analysis yet. Go to Home and process a video first.")
        return

    st.markdown('<div style="font-size:20px;font-weight:700;color:#E6EDF3;margin-bottom:4px;">💬 Chat with your Meeting</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#7D8590;margin-bottom:1rem;">Answers grounded in the actual transcript via RAG.</div>', unsafe_allow_html=True)

    rag_chain = result.get("rag_chain")

    if not st.session_state.chat_history:
        st.markdown('<div style="font-size:12px;font-weight:600;color:#7D8590;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.4px;">Suggested questions</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate([
            "What were the main topics discussed?",
            "Who is responsible for what?",
            "What decisions were made?",
            "What are the next steps?",
        ]):
            with cols[i % 2]:
                if st.button(q, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("Thinking…"):
                        answer = ask_question(rag_chain, q)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

    if st.session_state.chat_history:
        chat_html = '<div class="chat-wrap">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f'<div class="msg-user"><div><div class="bubble">{msg["content"]}</div></div></div>'
            else:
                chat_html += f'<div class="msg-bot"><div class="avatar">🤖</div><div><div class="bubble">{msg["content"]}</div></div></div>'
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_send = st.columns([6, 1])
        with col_inp:
            user_input = st.text_input("Question", placeholder="Ask anything about this meeting…", label_visibility="collapsed")
        with col_send:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("🤖 Thinking…"):
            answer = ask_question(rag_chain, user_input.strip())
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


# ── Page: Transcript ──────────────────────────────────────────────────────────
def page_transcript():
    result = st.session_state.result
    if not result:
        st.warning("No analysis yet. Go to Home and process a video first.")
        return

    transcript = result.get("transcript", "")
    st.markdown('<div style="font-size:20px;font-weight:700;color:#E6EDF3;margin-bottom:4px;">📄 Full Transcript</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button("⬇️ Download TXT", transcript, "transcript.txt", "text/plain", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="transcript-box">{transcript}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:12px;font-size:12px;color:#484F58;">📝 {len(transcript.split()):,} words · 🔤 {len(transcript):,} chars</div>', unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "home":       page_home()
elif page == "results":  page_results()
elif page == "chat":     page_chat()
elif page == "transcript": page_transcript()
