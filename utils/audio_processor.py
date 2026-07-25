import yt_dlp
import os
import re
import time
import subprocess
import math
import platform
from pathlib import Path
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# ── Load .env from project root ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(">>> audio_processor.py loaded: v7-hardened <<<")

# ── Cookie config ─────────────────────────────────────────────────────────────
_cookie_env = os.getenv("YTDLP_COOKIES_FILE", "").strip()
if _cookie_env:
    _path = Path(_cookie_env)
    COOKIES_FILE = str(_path if _path.is_absolute() else (BASE_DIR / _path).resolve())
else:
    COOKIES_FILE = None

# Browser cookies only work on Windows (Chrome profile lock issues on Linux)
COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip() or None
if platform.system() != "Windows":
    COOKIES_FROM_BROWSER = None

# ── Proxy config (NEW) ─────────────────────────────────────────────────────────
# Set YTDLP_PROXY in .env, e.g. http://user:pass@host:port
# This is the single biggest lever if cookies alone aren't solving cloud blocks —
# YouTube blocklists datacenter IP ranges regardless of cookie validity.
PROXY_URL = os.getenv("YTDLP_PROXY", "").strip() or None

# tv_embedded/mweb tend to survive PO-Token enforcement longest; web/android are
# the most frequently blocked on datacenter IPs as of 2026. Order = fail-fast to
# most-likely-to-work, not the reverse, to minimize wasted requests against
# YouTube (fewer requests = lower chance of a rate-limit/ban on this IP).
CLIENT_FALLBACK_ORDER = ["tv_embedded", "mweb", "ios", "android", "web"]

# Base delay (seconds) before each retry attempt; grows with each failure so we
# don't hammer YouTube back-to-back, which is itself a block trigger.
RETRY_BASE_DELAY = 3

print("=" * 60)
print(f"OS               : {platform.system()}")
print(f"Cookies File     : {COOKIES_FILE}")
print(f"Cookies Exists   : {os.path.exists(COOKIES_FILE) if COOKIES_FILE else False}")
print(f"Browser Cookies  : {COOKIES_FROM_BROWSER}")
print(f"Proxy Configured : {'yes' if PROXY_URL else 'no'}")
try:
    print(f"yt-dlp version   : {yt_dlp.version.__version__}")
except Exception:
    pass
print("=" * 60)


def _build_ydl_opts(output_path: str, player_client: str) -> dict:
    opts = {
        # Let yt-dlp pick the best audio itself — avoids a second probe request.
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "force_ipv4": True,
        "geo_bypass": True,
        "prefer_ffmpeg": True,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 3,
        "fragment_retries": 3,
        # Small randomized pacing between requests — reduces the "bot-like burst
        # traffic" signal that triggers 429s on cloud IPs.
        "sleep_interval_requests": 1,
        "sleep_interval": 1,
        "max_sleep_interval": 3,
        "extractor_args": {
            "youtube": {
                "player_client": [player_client],
                "skip": ["translated_subs"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    if PROXY_URL:
        opts["proxy"] = PROXY_URL
        print(f"  🌐 Using proxy: {PROXY_URL.split('@')[-1] if '@' in PROXY_URL else PROXY_URL}")

    # Priority 1: cookies.txt file
    if COOKIES_FILE and os.path.isfile(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
        print(f"  🍪 Using cookies file: {COOKIES_FILE}")
    # Priority 2: browser cookies (Windows only)
    elif COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER, None, None, None)
        print(f"  🍪 Using browser cookies: {COOKIES_FROM_BROWSER}")
    else:
        print("  ⚠️  No cookies — YouTube may block cloud/server IPs")

    return opts


def _is_drm_error(text: str) -> bool:
    return any(k in text.lower() for k in [
        "drm protected", "drm-protected", "widevine", "is encrypted"
    ])


def _is_auth_error(text: str) -> bool:
    return any(k in text.lower() for k in [
        "403", "http error 403", "sign in", "bot",
        "too many requests", "429", "login required", "use --cookies",
    ])


def _extract_video_id(url: str) -> str | None:
    match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def get_youtube_transcript(url: str, language: str = "english") -> str | None:
    """
    Free, low-risk first attempt: pull YouTube's own captions instead of
    downloading audio. This hits a different, much less aggressively
    blocked endpoint than yt-dlp's video download path — no proxy needed
    for most videos. Returns None (not an error) if no captions exist,
    so the caller can fall back to audio download.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        print("  ⚠️  Could not extract video ID for transcript lookup")
        return None

    lang_map = {"english": "en", "hinglish": "hi", "hindi": "hi"}
    preferred = lang_map.get(language.lower(), "en")

    try:
        print(f"  📝 Trying YouTube captions first (lang preference: {preferred})...")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript([preferred, "en"])
        except NoTranscriptFound:
            # fall back to whatever is available, even auto-generated
            transcript = next(iter(transcript_list))

        entries = transcript.fetch()
        text = " ".join(e["text"] for e in entries if e.get("text"))

        if text.strip():
            print(f"  ✅ Got transcript from captions ({len(text)} chars) — audio download skipped")
            return text

        print("  ⚠️  Captions were empty")
        return None

    except (TranscriptsDisabled, NoTranscriptFound):
        print("  ⚠️  No captions available for this video — falling back to audio download")
        return None
    except Exception as e:
        print(f"  ⚠️  Caption fetch failed ({e}) — falling back to audio download")
        return None


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    last_error = None
    auth_blocked = False

    for attempt, client in enumerate(CLIENT_FALLBACK_ORDER):
        if attempt > 0:
            delay = RETRY_BASE_DELAY * attempt
            print(f"\n  ⏳ Waiting {delay}s before next client (avoids burst-request blocks)...")
            time.sleep(delay)

        print(f"\n--- Trying player_client='{client}' ---")
        download_opts = _build_ydl_opts(output_path, client)

        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                base = os.path.splitext(ydl.prepare_filename(info))[0]
                filename = base + ".wav"

            if os.path.exists(filename):
                print(f"  ✅ Download success: {filename}")
                return filename

            # Fallback: scan downloads folder for newest .wav
            wav_files = [
                os.path.join(DOWNLOAD_DIR, f)
                for f in os.listdir(DOWNLOAD_DIR)
                if f.endswith(".wav")
            ]
            if wav_files:
                latest = max(wav_files, key=os.path.getmtime)
                print(f"  ✅ Found by scan: {latest}")
                return latest

            last_error = f"Download finished but WAV not found: {filename}"

        except yt_dlp.utils.DownloadError as e:
            err = str(e)
            print(f"  Download failed ({client}): {err[:300]}")
            if _is_drm_error(err):
                raise RuntimeError("This video is DRM-protected and cannot be downloaded.")
            if _is_auth_error(err):
                auth_blocked = True
            last_error = err
            continue

    # All clients exhausted
    if auth_blocked or (COOKIES_FILE is None and COOKIES_FROM_BROWSER is None and PROXY_URL is None):
        raise RuntimeError(
            "YouTube is blocking this download.\n\n"
            "This is an IP/authentication block — not a code error.\n\n"
            "Things to try, in order of impact:\n"
            "  1. Set YTDLP_PROXY in .env to a residential/rotating proxy —\n"
            "     this is usually what actually fixes cloud-IP blocks.\n"
            "  2. Confirm YTDLP_COOKIES_FILE points to a FRESH cookies.txt export\n"
            "     (cookies expire after some weeks — re-export if old).\n"
            "  3. Run: pip install -U yt-dlp  (YouTube changes detection often).\n"
            "  4. As a last resort, use the app's Upload File tab: download the\n"
            "     video locally and upload the file directly.\n\n"
            f"Technical detail: {last_error}"
        )

    raise RuntimeError(
        f"Download failed after trying all clients.\n"
        f"Last error: {last_error}\n"
        f"Try: pip install -U yt-dlp"
    )


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to mono 16kHz WAV."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path],
        check=True,
    )
    return output_path


def get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    return float(result.stdout.strip())


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    duration = get_audio_duration(wav_path)
    chunk_seconds = chunk_minutes * 60
    total_chunks = math.ceil(duration / chunk_seconds)
    chunks = []
    base = os.path.splitext(wav_path)[0]

    for i in range(total_chunks):
        chunk_path = f"{base}_chunk_{i}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-ss", str(i * chunk_seconds),
            "-t", str(chunk_seconds),
            "-ac", "1", "-ar", "16000", chunk_path,
        ], check=True)
        chunks.append(chunk_path)

    return chunks


def process_input(source: str, language: str = "english") -> dict:
    """
    Returns a dict:
      {"type": "text", "transcript": "..."}          — captions found, no audio needed
      {"type": "chunks", "chunks": [...]}             — audio path, needs transcription
    """
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Trying captions first (free, low block-risk)...")
        transcript = get_youtube_transcript(source, language)
        if transcript:
            return {"type": "text", "transcript": transcript}

        print("No captions available. Falling back to audio download...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return {"type": "chunks", "chunks": chunks}
