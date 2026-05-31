# YouTube Shorts Pipeline

YouTube Shorts Pipeline is a Streamlit tool that analyzes uploaded video, generates Turkish narration, creates ElevenLabs voiceover, produces ASS/SRT subtitles, and renders final Shorts with FFmpeg.

## Snapshot

- **Category:** AI media automation
- **Status:** Public repository
- **Repository:** https://github.com/yusufky63/youtube-shorts-video-pipeline
- **Portfolio:** https://codexsha.dev

## Product Scope

YouTube Shorts Pipeline is documented here as a product repository, not just a code dump. The goal of this README is to make the product purpose, runtime surface, and development path clear for future review and maintenance.

## Core Capabilities

- Gemini-based video analysis and script generation
- ElevenLabs TTS voiceover generation
- ASS/SRT subtitle output
- FFmpeg final MP4 render pipeline
- Streamlit operator interface

## Existing README Coverage Preserved

This refresh keeps the important project-specific areas from the previous documentation:

- Calistirma
- Ana Ozellikler
- Notlar

## Tech Stack

- Python
- Streamlit
- Google Gemini
- ElevenLabs
- FFmpeg
- imageio-ffmpeg
- pytest

## Repository Map

| Path | Purpose |
| --- | --- |
| ui/streamlit_app.py | Main Streamlit interface |
| requirements.txt | Python dependencies |
| tests/ | Tests where present |
| output/ | Generated local media outputs, if configured locally |

## Local Development

| Command | Purpose |
| --- | --- |
| python -m venv venv | Create virtual environment |
| .\venv\Scripts\activate | Activate on Windows PowerShell |
| pip install -r requirements.txt | Install dependencies |
| streamlit run ui\streamlit_app.py | Run the Streamlit UI |
| pytest | Run tests where available |

## Environment Notes

Use local environment files for secrets and deployment-specific values. Do not commit real keys.

- Gemini API key
- ElevenLabs API key
- Local FFmpeg/imageio-ffmpeg availability

## Operational Notes

- Keep this README aligned with the live product and portfolio copy.
- Prefer small, documented changes over large undocumented rewrites.
- The previous README was Turkish and concise. This version keeps the same run flow but documents the full pipeline more clearly.

## Maintainer

Built by Yusuf / Codexsha.

- GitHub: https://github.com/yusufky63
- X: https://x.com/codexsha
- Telegram: https://t.me/codexsha
