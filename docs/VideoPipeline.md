# Video Pipeline

This pipeline takes an MP4 video, asks Gemini to create timed Turkish narration,
generates Edge TTS audio for each segment, writes an SRT file, and renders a final
MP4 with hard subtitles.

## Folder Structure

```text
src/video_pipeline/
  cli.py              # CLI argument parsing
  pipeline.py         # End-to-end orchestration
  gemini_analyzer.py  # Gemini upload, analysis, and JSON parsing
  json_parser.py      # Gemini JSON validation
  srt_writer.py       # SRT file generation
  tts_generator.py    # Edge TTS MP3 generation
  ffmpeg_renderer.py  # FFmpeg audio mix and hard subtitles
  timecode.py         # SRT timestamp helpers
  models.py           # Shared dataclasses
scripts/video_pipeline.py
scripts/run_video_pipeline_ui.py
ui/streamlit_app.py   # Browser UI for managing jobs
video_pipeline_runs/  # Generated files, ignored by git
```

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

FFmpeg can be installed system-wide, or the app can use the `imageio-ffmpeg`
Python fallback from `requirements.txt`.

System-wide check:

```bash
ffmpeg -version
```

Set your Gemini API key:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-key"

# macOS/Linux
export GEMINI_API_KEY="your-key"
```

Optional ElevenLabs TTS:

```bash
# Windows PowerShell
$env:ELEVENLABS_API_KEY="your-key"

# macOS/Linux
export ELEVENLABS_API_KEY="your-key"
```

## Usage

```bash
python scripts/video_pipeline.py path/to/input.mp4
```

Optional example:

```bash
python scripts/video_pipeline.py path/to/input.mp4 --model gemini-2.5-flash --voice tr-TR-AhmetNeural --original-volume 0.30
```

The default output is:

```text
video_pipeline_runs/final_output.mp4
```

## Browser UI

Run the Streamlit UI from the repository root:

```bash
streamlit run ui/streamlit_app.py
```

or:

```bash
python scripts/run_video_pipeline_ui.py
```

The UI supports:

- MP4 upload and per-job folders under `video_pipeline_runs/`
- Gemini prompt/model/API key settings
- Gemini model refresh from `ListModels` for API-key-specific availability
- editable narration table before SRT generation
- Edge TTS voice/rate/volume settings
- optional ElevenLabs TTS provider with API key, voice ID, model, and voice settings
- selected voice preview before generating all TTS files
- TTS segment previews
- subtitle style presets
- FFmpeg availability check with Python fallback support
- output paths panel for final MP4, JSON, SRT, ASS, and TTS files
- final FFmpeg render and MP4 download

Generated files are stored per job:

```text
video_pipeline_runs/<timestamp-video-name>/
  input.mp4
  narration.json
  narration.srt
  narration.ass
  tts/
  final_output.mp4
```
