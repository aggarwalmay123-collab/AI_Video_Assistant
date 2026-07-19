import yt_dlp

from dotenv import load_dotenv

load_dotenv()

import os
import subprocess
import math

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(">>> audio_processor.py loaded: v4-no-oauth <<<")
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

cookie_env = os.getenv("YTDLP_COOKIES_FILE")

if cookie_env:
    COOKIES_FILE = str((BASE_DIR / cookie_env).resolve())
else:
    COOKIES_FILE = None
    print("Cookie path:", COOKIES_FILE)
print("Exists:", os.path.exists(COOKIES_FILE) if COOKIES_FILE else False)




        # cookies.txt path
COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")  # e.g. "chrome"

CLIENT_FALLBACK_ORDER = ["mweb", "android", "ios", "web", "tv_embedded"]

print("COOKIES_FILE =", repr(COOKIES_FILE))
print("COOKIES_FROM_BROWSER =", repr(COOKIES_FROM_BROWSER))
print("File exists:", os.path.exists(COOKIES_FILE) if COOKIES_FILE else False)



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
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
        print(f"  Using cookies file: {COOKIES_FILE}")

    # Priority 2: cookies from browser (chrome/firefox/edge/brave)
    elif COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER, None, None, None)
        print(f"  Using cookies from browser: {COOKIES_FROM_BROWSER}")

    else:
        print("  ⚠️  No cookies configured — YouTube may block. Set YTDLP_COOKIES_FILE or YTDLP_COOKIES_FROM_BROWSER in .env")

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
        print(f"  Selected audio-only: {best['format_id']} ({best.get('ext','?')}, {best.get('abr','?')}kbps)")
        return best["format_id"]

    progressive = [
        f for f in formats
        if f.get("acodec") not in (None, "none")
        and f.get("vcodec") not in (None, "none")
    ]
    if progressive:
        progressive.sort(key=lambda f: f.get("height") or 0)
        best = progressive[0]
        print(f"  Selected progressive: {best['format_id']} ({best.get('ext','?')}, {best.get('height','?')}p)")
        return best["format_id"]

    return None


def _is_drm_error(error_text: str) -> bool:
    drm_keywords = ["drm protected", "drm-protected", "widevine", "is encrypted"]
    return any(kw in error_text.lower() for kw in drm_keywords)


def _is_auth_error(error_text: str) -> bool:
    auth_keywords = [
        "sign in to confirm", "bot", "429", "too many requests",
        "http error 403", "403", "login required", "oauth",
        "use --cookies", "cookies",
    ]
    return any(kw in error_text.lower() for kw in auth_keywords)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    last_error = None

    for client in CLIENT_FALLBACK_ORDER:
        print(f"\n--- Trying player_client='{client}' ---")

        probe_opts = _build_ydl_opts(output_path, client, "bestaudio/best")
        probe_opts["quiet"] = True
        probe_opts["no_warnings"] = True

        try:
            print(f"  Probing formats...")
            with yt_dlp.YoutubeDL(probe_opts) as ydl:
                info = ydl.extract_info(url, download=False)

        except yt_dlp.utils.DownloadError as e:
            error_text = str(e)
            print(f"  Probe failed ({client}): {error_text[:200]}")

            if _is_drm_error(error_text):
                raise RuntimeError(
                    f"This video is genuinely DRM-protected and cannot be downloaded."
                ) from e

            if _is_auth_error(error_text):
                last_error = RuntimeError(
                    "YouTube is blocking downloads (authentication required).\n\n"
                    "QUICK FIX — add ONE of these to your .env file:\n\n"
                    "Option A (recommended): Use browser cookies directly\n"
                    "  YTDLP_COOKIES_FROM_BROWSER=chrome\n"
                    "  (or: firefox / edge / brave / opera)\n\n"
                    "Option B: Export cookies.txt manually\n"
                    "  1. Install 'Get cookies.txt LOCALLY' Chrome extension\n"
                    "  2. Go to youtube.com while logged in\n"
                    "  3. Export → save as cookies.txt in project root\n"
                    "  4. Add: YTDLP_COOKIES_FILE=cookies.txt to .env\n\n"
                    f"Raw error: {error_text[:300]}"
                )
                continue

            last_error = e
            continue

        formats = info.get("formats") or []
        print(f"  Got {len(formats)} formats")
        format_id = _pick_best_format(formats)

        if not format_id:
            print(f"  No usable formats for client='{client}'")
            last_error = RuntimeError(f"No usable formats from client '{client}'.")
            continue

        download_opts = _build_ydl_opts(output_path, client, format_id)

        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                base = os.path.splitext(ydl.prepare_filename(info))[0]
                filename = base + ".wav"

            if os.path.exists(filename):
                print(f"  ✅ Success! File: {filename}")
                return filename

            last_error = RuntimeError(f"File not found after download: {filename}")

        except yt_dlp.utils.DownloadError as e:
            print(f"  Download failed ({client}): {e}")
            last_error = e
            continue

    raise RuntimeError(str(last_error))


def convert_to_wav(input_path: str) -> str:
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
            "-ss", str(i * chunk_seconds), "-t", str(chunk_seconds),
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