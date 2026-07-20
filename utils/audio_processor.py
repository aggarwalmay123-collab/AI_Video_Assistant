import yt_dlp
import os
import subprocess
import math
import platform
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(">>> audio_processor.py loaded: v6-clean <<<")

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

CLIENT_FALLBACK_ORDER = ["tv_embedded", "mweb", "android", "ios", "web"]

print("=" * 60)
print(f"OS               : {platform.system()}")
print(f"Cookies File     : {COOKIES_FILE}")
print(f"Cookies Exists   : {os.path.exists(COOKIES_FILE) if COOKIES_FILE else False}")
print(f"Browser Cookies  : {COOKIES_FROM_BROWSER}")
print("=" * 60)


def _build_ydl_opts(output_path: str, player_client: str, format_selector: str) -> dict:
    opts = {
        "format": format_selector,
        "outtmpl": output_path,
        "force_ipv4": True,
        "geo_bypass": True,
        "prefer_ffmpeg": True,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 5,
        "fragment_retries": 5,
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


def _pick_best_format(formats: list):
    if not formats:
        return None

    audio_only = [
        f for f in formats
        if f.get("acodec") not in (None, "none")
        and f.get("vcodec") in (None, "none")
    ]
    if audio_only:
        audio_only.sort(key=lambda f: f.get("abr") or 0, reverse=True)
        best = audio_only[0]
        print(f"  ✔ Audio-only: {best['format_id']} ({best.get('ext','?')}, {best.get('abr','?')}kbps)")
        return best["format_id"]

    progressive = [
        f for f in formats
        if f.get("acodec") not in (None, "none")
        and f.get("vcodec") not in (None, "none")
    ]
    if progressive:
        progressive.sort(key=lambda f: f.get("height") or 0)
        best = progressive[0]
        print(f"  ✔ Progressive: {best['format_id']} ({best.get('ext','?')}, {best.get('height','?')}p)")
        return best["format_id"]

    return None


def _is_drm_error(text: str) -> bool:
    return any(k in text.lower() for k in [
        "drm protected", "drm-protected", "widevine", "is encrypted"
    ])


def _is_auth_error(text: str) -> bool:
    return any(k in text.lower() for k in [
        "403", "http error 403", "sign in", "bot",
        "too many requests", "429", "login required", "use --cookies",
    ])


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    last_error = None
    auth_blocked = False

    for client in CLIENT_FALLBACK_ORDER:
        print(f"\n--- Trying player_client='{client}' ---")

        # Step 1: probe available formats
        probe_opts = _build_ydl_opts(output_path, client, "bestaudio/best")
        probe_opts["quiet"] = True
        probe_opts["no_warnings"] = True

        try:
            print("  Probing formats...")
            with yt_dlp.YoutubeDL(probe_opts) as ydl:
                info = ydl.extract_info(url, download=False)

        except yt_dlp.utils.DownloadError as e:
            err = str(e)
            print(f"  Probe failed: {err[:300]}")
            if _is_drm_error(err):
                raise RuntimeError("This video is DRM-protected and cannot be downloaded.")
            if _is_auth_error(err):
                auth_blocked = True
            last_error = err
            continue

        formats = info.get("formats") or []
        print(f"  Got {len(formats)} formats")
        format_id = _pick_best_format(formats)

        if not format_id:
            print(f"  No usable formats for '{client}'")
            last_error = f"No usable formats from client '{client}'"
            continue

        # Step 2: download with confirmed format_id
        download_opts = _build_ydl_opts(output_path, client, format_id)

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
            if _is_auth_error(err):
                auth_blocked = True
            last_error = err
            continue

    # All clients exhausted
    if auth_blocked or (COOKIES_FILE is None and COOKIES_FROM_BROWSER is None):
        raise RuntimeError(
            "YouTube is blocking this download.\n\n"
            "This is an IP/authentication block — not a code error.\n\n"
            "SOLUTION (use the Upload File tab in the app):\n"
            "  1. Download the video on your local PC using yt-dlp or any tool\n"
            "  2. Go to the app → Upload File tab\n"
            "  3. Upload the MP4/MP3/WAV file directly\n\n"
            "OR for local use, add to .env:\n"
            "  YTDLP_COOKIES_FROM_BROWSER=chrome\n\n"
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


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
