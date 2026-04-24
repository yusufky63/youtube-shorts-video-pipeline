# youtube-shorts-video-pipeline

Streamlit tabanli YouTube Shorts video pipeline araci.

Bu proje yuklenen videoyu analiz eder, Turkce anlatim uretir, ElevenLabs TTS ile seslendirir, ASS/SRT altyazi olusturur ve FFmpeg ile final MP4 render alir.

## Calistirma

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui\streamlit_app.py
```

## Ana Ozellikler

- Gemini ile video analizi ve senaryo uretimi
- ElevenLabs TTS entegrasyonu
- Ses ve altyazi senkronizasyonu
- Kelime kelime karaoke altyazi
- 3D golge, kalin outline ve Shorts caption stilleri
- Logo, crop, blur arka plan ve sansur bolgesi destegi
- YouTube baslik, aciklama ve etiket ciktilari

## Notlar

- FFmpeg sistemde kurulu olmali veya `imageio-ffmpeg` uzerinden bulunabilmeli.
- API anahtarlari UI icinden yerel olarak kaydedilebilir.
- Uretilen videolar `video_pipeline_runs/` altinda tutulur ve git'e eklenmez.
