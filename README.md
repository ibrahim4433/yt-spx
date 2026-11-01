
## 📂 Repository Structure

```
yt-ch/
├── all_channels_videos.csv            # Original source metadata for 2,640 videos
├── download_videos.py                 # Subtitle fetcher and cleaner via yt-dlp
├── run_app.py                         # One-click launcher for the web application
├── dawah_knowledge_extractor_colab.ipynb # Free-tier Google Colab batch extraction notebook
│
├── core/
│   ├── srt_parser.py                  # Parses subtitle cues, removes duplicates & chunks dialogues
│   ├── metadata_linker.py             # Links SRT files to Arabic titles and YouTube URLs
│   ├── extractor_prompt.py            # Standardized schema for dialectical arguments & analogies
│   ├── colab_runner.py                # Checkpointed batch processor for Google Colab
│   └── smart_bot.py                   # Zero-LLM intelligent dialogue assistant
│
├── db/
│   └── database.py                    # SQLite engine with FTS5 dual-language BM25 search
│
├── data/
│   ├── video_manifest.json            # Unified inventory of 2,363 linked videos
│   ├── pilot_qa_dataset.json          # Pre-analyzed dialectical Q&A units across 6 pillars
│   └── dawah_knowledge.db             # SQLite database with FTS5 search index
│
├── web/
│   ├── app.py                         # FastAPI REST application
│   ├── templates/
│   │   └── index.html                 # Responsive bilingual web portal
│   └── static/
│       ├── css/style.css              # Dark slate theme, RTL/LTR styling & card layouts
│       └── js/app.js                  # Dynamic search, language switcher & chat client
│
└── tests/
    └── test_endpoints.py              # Automated test suite for all endpoints
```
