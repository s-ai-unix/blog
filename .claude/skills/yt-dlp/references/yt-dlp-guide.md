# yt-dlp Usage Guide

## Common Format Options

- `bestvideo+bestaudio/best` - Best quality video + audio merged
- `bestvideo[height<=1080]+bestaudio/best` - Max 1080p
- `bestvideo[height<=720]+bestaudio/best` - Max 720p
- `bestaudio/best` - Best quality audio only

## Subtitle Options

- `--sub-lang en,zh-Hans,zh-Hant` - Download specific subtitle languages
- `--subs all` - Download all available subtitles
- `--embed-subs` - Embed subtitles into video file (MP4/MKV only)
- `-k` - Keep subtitle files separately

## Thumbnail Options

- `--write-thumbnail` - Download thumbnail
- `--convert-thumbnails jpg` - Convert to JPG format
- `--embed-thumbnail` - Embed thumbnail as cover art (audio files)

## Playlist Options

- `--no-playlist` - Download only single video from playlist URL
- `--playlist-start 1` - Start from index
- `--playlist-end 10` - End at index
- `--playlist-items 1,3,5,7-10` - Download specific items

## Output Templates

- `%(title)s.%(ext)s` - Video title
- `%(uploader)s/%(title)s.%(ext)s` - Organize by uploader
- `%(playlist_index)s-%(title)s.%(ext)s` - Include playlist index
- `%(upload_date)s-%(title)s.%(ext)s` - Include upload date

## Audio Extraction

- `--extract-audio` - Extract audio track
- `--audio-format mp3` - Convert to MP3 (also: m4a, wav, flac)
- `--audio-quality 0` - Best quality (0-9, lower is better)

## Post-Processing

- `--merge-output-format mp4` - Ensure final output is MP4
- `--convert-srts srt` - Convert subtitles to SRT format
- `--remux-video mp4` - Remux to MP4 container

## Info Extraction

- `--dump-json` - Output full JSON info
- `--list-formats` - List all available formats
- `--list-subs` - List available subtitles
- `--write-info-json` - Write video info to JSON file

## Common Issues

**1. "ffmpeg not found"**
- Install FFmpeg for merging video/audio
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

**2. "Sign in to confirm you're not a bot"**
- Use `--cookies-from-browser chrome` to use browser cookies
- Or download cookies.txt and use `--cookies file.txt`

**3. "Video unavailable"**
- Try with `--check-formats` to check format availability
- Some regions may require VPN or specific cookies

**4. Download speed slow**
- Use `--concurrent-fragments 4` for parallel downloads
- Use `--limit-rate 1M` to limit bandwidth

## Supported Sites

yt-dlp supports 1000+ sites including:
- YouTube (all variants)
- Bilibili
- Vimeo
- Twitter/X
- TikTok
- Instagram
- Facebook
- Twitch
- And many more...

Check full list: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md
