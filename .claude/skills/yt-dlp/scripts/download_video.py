#!/usr/bin/env python3
"""
yt-dlp video downloader script
Supports downloading videos, audio, subtitles, and thumbnails from 1000+ websites
"""

import sys
import json
import subprocess
from pathlib import Path


def download_video(
    url,
    output_dir="~/Downloads/videos",
    format="bestvideo+bestaudio/best",
    subtitles=False,
    embed_subs=False,
    thumbnail=False,
    audio_only=False,
    quality=None,
    playlist=False,
    start=None,
    end=None
):
    """
    Download video using yt-dlp

    Args:
        url: Video URL
        output_dir: Output directory (supports ~ expansion)
        format: Video format selection
        subtitles: Download subtitles
        embed_subs: Embed subtitles in video
        thumbnail: Download thumbnail
        audio_only: Download audio only
        quality: Video quality (e.g., "1080", "720")
        playlist: Download entire playlist
        start: Playlist start index (1-based)
        end: Playlist end index
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = ["yt-dlp", "-o", str(output_path / "%(title)s.%(ext)s")]

    # Format selection
    if audio_only:
        cmd.extend(["-f", "bestaudio/best", "--extract-audio", "--audio-format", "mp3"])
    elif quality:
        cmd.extend(["-f", f"bestvideo[height<={quality}]+bestaudio/best"])
    else:
        cmd.extend(["-f", format])

    # Subtitles
    if subtitles:
        cmd.extend(["--sub-lang", "en,zh-Hans,zh-Hant", "--subs", "all", "-k"])
        if embed_subs:
            cmd.append("--embed-subs")

    # Thumbnail
    if thumbnail:
        cmd.extend(["--write-thumbnail", "--convert-thumbnails", "jpg"])

    # Playlist options
    if not playlist:
        cmd.append("--no-playlist")
    else:
        if start:
            cmd.extend(["--playlist-start", str(start)])
        if end:
            cmd.extend(["--playlist-end", str(end)])

    # Progress bar
    cmd.extend(["--newline", "--progress"])

    # URL
    cmd.append(url)

    # Execute
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return {
            "success": True,
            "output": result.stdout,
            "error": result.stderr,
            "path": str(output_path)
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr,
            "output": e.stdout
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "yt-dlp not found. Install with: pip install yt-dlp"
        }


def get_info(url):
    """Get video information without downloading"""
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", url],
            check=True,
            capture_output=True,
            text=True
        )
        info = json.loads(result.stdout)
        return {
            "success": True,
            "info": info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_video.py <url> [options]")
        sys.exit(1)

    url = sys.argv[1]
    result = download_video(url)

    if result["success"]:
        print(f"✅ Downloaded to: {result['path']}")
    else:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
