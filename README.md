# youtube-shorts-video-pipeline

YouTube Shorts Video Pipeline, uzun videolardan veya kaynak medyadan Shorts formatinda ciktilar uretmek icin hazirlanmis Python ve Streamlit tabanli bir otomasyon aracidir.

Pipeline; video analizi, Turkce anlatim metni, ElevenLabs seslendirme, ASS/SRT altyazi ve FFmpeg render adimlarini tek akista toplar.

## Calistirma

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui\streamlit_app.py
```

CLI veya script tabanli akislarda `scripts/` ve `src/` altindaki yardimci dosyalar kullanilabilir.

## Ana Ozellikler

- Video veya sahne analizi icin Gemini destekli akil yurutme akisi.
- Turkce Shorts anlatim metni olusturma.
- ElevenLabs ile voiceover uretimi.
- ASS ve SRT altyazi uretimi.
- FFmpeg ile dikey Shorts render pipeline.
- Streamlit UI ile daha kolay manuel kontrol.

## Proje Yapisi

- `ui/streamlit_app.py` - Streamlit arayuzu.
- `src/` - pipeline, analiz, altyazi, ses ve render mantigi.
- `scripts/` - yardimci calistirma/otomasyon scriptleri.
- `docs/` - notlar ve ek dokumantasyon.
- `tests/` - test dosyalari.
- `requirements.txt` - Python bagimliliklari.

## Teknoloji

| Katman | Araclar |
| --- | --- |
| UI | Streamlit |
| AI analiz | Google Gemini |
| Ses | ElevenLabs |
| Altyazi | ASS, SRT |
| Render | FFmpeg |
| Dil | Python |

## Notlar

- API anahtarlarini `.env` veya lokal ortam degiskenlerinde tutun; repoya secret commit etmeyin.
- Render kalitesi FFmpeg presetleri, kaynak video cozunurlugu ve altyazi stiline gore degisir.
- Cikti dosyalarini buyuk medya dosyalariyla beraber versiyon kontrolune eklememek daha sagliklidir.

## Status

- Repository: https://github.com/yusufky63/youtube-shorts-video-pipeline
