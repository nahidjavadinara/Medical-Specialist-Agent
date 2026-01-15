# 🏥 Medical AI Agent – Hybrid Architecture (SLM + OpenAI)

A production-ready medical records analysis system supporting **Small Language Models (SLM)**, **Large Language Models (LLM via OpenAI)**, and **Hybrid (SLM + LLM)** workflows with automatic fallback and quota-safe execution.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents
- [Features](#features)
- [Supported File Formats](#supported-file-formats)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture Modes](#architecture-modes)
- [System Requirements](#system-requirements)
- [Troubleshooting](#troubleshooting)
- [Security & Compliance](#security--compliance)

---

## ✨ Features

### 🎯 Multi-Format Medical Record Ingestion
- **Documents**: PDF, DOCX, XLSX
- **Images**: PNG, JPG, JPEG (OCR via Tesseract)
- **Audio**: MP3, WAV (speech-to-text)
- **Video**: MP4, AVI, MOV (audio extraction + transcription)

### 🤖 Flexible AI Architectures
- **SLM Mode** – Fully local analysis using **FLAN-T5-Large**
- **LLM Mode** – Cloud-based analysis using **OpenAI gpt-4o-mini**
- **Hybrid Mode** – Local SLM + OpenAI synthesis with automatic fallback

### 📊 Medical Analysis Capabilities
- Patient summaries
- Potential diagnosis identification
- Treatment considerations
- Risk factor extraction
- Free-form medical Q&A

### 🔒 Privacy-Aware Design
- Local-only execution available (SLM)
- Cloud calls optional and explicit
- No persistent cloud storage
- Encrypted API communication (TLS)

---

## 📁 Supported File Formats

| Category | Formats | Processing |
|--------|--------|-----------|
| Documents | PDF, DOCX, XLSX | Text extraction |
| Images | PNG, JPG, JPEG | OCR (Tesseract) |
| Audio | MP3, WAV | Speech recognition |
| Video | MP4, AVI, MOV | Audio extraction + transcription |

---

## 🚀 Installation

### System Dependencies

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr ffmpeg
```

**macOS**
```bash
brew install tesseract ffmpeg
```

**Windows**
- https://github.com/UB-Mannheim/tesseract/wiki
- https://ffmpeg.org/download.html

---

### Python Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate       # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

### OpenAI Configuration (LLM / Hybrid only)

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Get your key from:
https://platform.openai.com/account/api-keys

> SLM mode does **not** require an API key.

---

## 💻 Usage

```bash
streamlit run medical_ai_app.py
```

Open:
http://localhost:8501

---

## 🏗️ Architecture Modes

### SLM (Local)
- Offline
- No cost
- Maximum privacy

### LLM (OpenAI)
- Requires billing
- Higher reasoning
- Internet required

### Hybrid (Recommended)
- SLM first, OpenAI refinement
- Automatic fallback on quota exhaustion
- No crashes

---

## 💾 System Requirements

**Minimum**
- 4 CPU cores
- 8 GB RAM
- Python 3.9+

**Recommended**
- 8+ cores
- 16 GB RAM
- Optional CUDA GPU

---

## 🔧 Troubleshooting

### OpenAI 429 / Quota exceeded
- Enable billing
- Switch to SLM
- Hybrid mode auto-fallbacks

### MoviePy
Use MoviePy 2.x syntax:
```python
from moviepy import VideoFileClip
```

---

## 🔒 Security & Compliance

- Designed with HIPAA / GDPR principles
- Not FDA-approved
- Research & decision-support only

---

## ⚠️ Disclaimer

This software is **not** a medical device.
Always consult qualified healthcare professionals.

---

## 📝 License

MIT License with Healthcare Addendum.

---

**Version**: 1.1.0  
**Last Updated**: January 2026
