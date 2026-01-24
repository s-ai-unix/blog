---
name: yt-dlp
description: Download videos, audio, subtitles, and thumbnails from 1000+ websites including YouTube, Bilibili, Vimeo, Twitter, TikTok, Instagram, and more. Use when user asks to download videos, extract audio, save content from video platforms, or mentions video links. Triggers on phrases like "download this video", "save from YouTube", "extract audio", "get video subtitles", or provides any video URL.
---

# yt-dlp Video Downloader

## Quick Start

Download a video with default settings (best quality, saves to `~/Downloads/videos`):

```bash
python3 scripts/download_video.py <video-url>
```

## Common Workflows

### Download Single Video

Default downloads best quality video + audio merged:

```bash
python3 scripts/download_video.py "https://www.youtube.com/watch?v=xxxxx"
```

### Download with Specific Quality

Limit to 720p:

```bash
python3 scripts/download_video.py "<url>" --quality 720
```

### Download Audio Only

Extract MP3 audio:

```bash
python3 scripts/download_video.py "<url>" --audio-only
```

### Download with Subtitles

Download and embed subtitles:

```bash
python3 scripts/download_video.py "<url>" --subtitles --embed-subs
```

### Download Playlist

Download entire playlist:

```bash
python3 scripts/download_video.py "<playlist-url>" --playlist
```

Download specific playlist range:

```bash
python3 scripts/download_video.py "<url>" --playlist --start 1 --end 5
```

### Get Video Info

Get video information without downloading:

```python
import json
from scripts.download_video import get_info

result = get_info("<url>")
if result["success"]:
    info = result["info"]
    print(f"Title: {info.get('title')}")
    print(f"Duration: {info.get('duration')}s")
    print(f"Available formats: {len(info.get('formats', []))}")
```

## Script Parameters

The `download_video.py` script accepts these parameters:

- `url` (required): Video URL
- `output_dir`: Output directory (default: `~/Downloads/videos`)
- `format`: Format selector (default: `bestvideo+bestaudio/best`)
- `subtitles`: Download subtitles (default: False)
- `embed_subs`: Embed subtitles in video (default: False)
- `thumbnail`: Download thumbnail (default: False)
- `audio_only`: Audio only mode (default: False)
- `quality`: Max quality height like "1080", "720" (default: None)
- `playlist`: Download full playlist (default: False)
- `start`: Playlist start index (default: None)
- `end`: Playlist end index (default: None)

## Interactive Usage

When user provides a video URL, ask clarifying questions:

1. **Single video or playlist?**
   - Single: Use default (no `--playlist`)
   - Playlist: Add `--playlist`, ask if they want specific range

2. **Quality preference?**
   - Best quality: No `--quality` flag
   - Specific quality: Use `--quality 720/1080/480`

3. **Additional content?**
   - Audio only: `--audio-only`
   - Subtitles: `--subtitles` (ask if they want to embed: `--embed-subs`)
   - Thumbnail: `--thumbnail`

4. **Custom save location?**
   - Default `~/Downloads/videos` or ask for custom path

## Advanced Options

For advanced options not covered by the script, use yt-dlp directly:

```bash
# List available formats
yt-dlp --list-formats <url>

# Download with custom output template
yt-dlp -o "%(uploader)s/%(title)s.%(ext)s" <url>

# Use browser cookies for restricted content
yt-dlp --cookies-from-browser chrome <url>

# Download specific format by ID
yt-dlp -f "137+140" <url>
```

See [references/yt-dlp-guide.md](references/yt-dlp-guide.md) for complete yt-dlp options.

## Requirements

Install yt-dlp and FFmpeg:

```bash
# Install yt-dlp
pip install yt-dlp

# Install FFmpeg (required for merging video/audio)
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Update yt-dlp regularly
pip install -U yt-dlp
```

## Troubleshooting

**"ffmpeg not found"**: Install FFmpeg using the commands above

**"Sign in to confirm"**: Use cookies with `yt-dlp --cookies-from-browser chrome <url>`

**Slow downloads**: Try `--concurrent-fragments 4` for parallel fragment downloads

**Region-restricted content**: May need VPN or cookies from browser in that region

## Return Value

The script returns a dict:

```python
{
    "success": True/False,
    "output": "stdout content",
    "error": "stderr content (if any)",
    "path": "/path/to/output/directory"  # on success
}
```
