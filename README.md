# YouTube Shorts Pipeline

![Category](https://img.shields.io/badge/Category-AI%20Media%20Automation-1f1f1f?style=flat-square&labelColor=141414&color=2b2b2b) ![Status](https://img.shields.io/badge/Status-public-1f1f1f?style=flat-square&labelColor=141414&color=2b2b2b)

Streamlit video pipeline for analysis, Turkish narration, ElevenLabs voiceover, ASS/SRT subtitles, and FFmpeg rendering.

## Links

- Repository: https://github.com/yusufky63/youtube-shorts-video-pipeline
- Portfolio: https://codexsha.dev

## Overview

YouTube Shorts Pipeline is part of the Codexsha product portfolio. The project is focused on shipping a compact, usable product surface rather than a demo-only prototype. This README is written to make the repository easier to understand, run, and evaluate.

## Key Features

- Video analysis and script generation
- ElevenLabs voiceover generation
- ASS/SRT subtitle generation
- FFmpeg Shorts rendering pipeline
- Streamlit operator UI

## Stack

- Python
- Streamlit
- Google Gemini
- ElevenLabs
- FFmpeg

## Role / Ownership

Built the video processing workflow, AI narration path, subtitle pipeline, and render automation.

## Getting Started

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui\streamlit_app.py
```

## Environment

Create a local environment file from the project conventions and configure only the values needed for the flow you are running. Do not commit secrets.

Typical values used by this project include:

- Gemini API key
- ElevenLabs API key
- local FFmpeg/imageio-ffmpeg availability

## Project Notes

- Status: Public repository.
- Private or sensitive implementation details are intentionally not documented in public-facing copy.
- The README should stay aligned with the live product and the Codexsha portfolio page.

## Maintainer

Built by Yusuf / Codexsha.

- GitHub: https://github.com/yusufky63
- X: https://x.com/codexsha
- Telegram: https://t.me/codexsha
