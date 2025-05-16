# Music Analysis Suite 🎵

![Project Banner](https://example.com/path/to/banner-image.jpg) *<!-- Replace with actual image URL -->*


## Live Preview
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://musicllm.streamlit.app/)

## Overview

A comprehensive AI-powered platform for music analysis, virality prediction, and expert recommendations. This suite combines audio processing, machine learning, and large language models to provide actionable insights for musicians, producers, and music industry professionals.

## Key Features

- **Instrument Detection**: Identifies and quantifies musical instruments in audio files
- **Virality Prediction**: Predicts a song's potential popularity using 14+ audio features
- **AI Music Consultant**: Groq-powered LLM provides tailored recommendations
- **Interactive Dashboard**: Visualizations and chat interface in one workspace
- **Professional Workflow**: Three-step analysis process (Upload → Predict → Consult)

## Technology Stack

### Core Components
| Component          | Technology               |
|--------------------|--------------------------|
| Audio Analysis     | TensorFlow Hub (YAMNet)  |
| Virality Model     | Keras Neural Network     |
| AI Consultant      | Groq (LLaMA3-8b)         |
| Backend Framework  | Streamlit                |
| Visualization      | Matplotlib/Seaborn       |

### Dependencies
# Core dependencies
streamlit==1.34.0
numpy==1.26.4
pandas==2.2.1
librosa==0.10.1
ensorflow==2.16.1
tensorflow-hub==0.16.1
noisereduce==1.0.0
# Machine Learning/Data Processing
scikit-learn==1.6.0
joblib==1.3.2
# LLM and LangChain
langchain-groq==0.1.2
langchain-core==0.1.33
# Audio processing
soundfile==0.12.1
resampy==0.4.2
# Utilities
python-dotenv==1.0.1
tqdm==4.66.2

## Dataset
Spotify_1Million_Tracks: https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks
## Installation
# Prerequisites
- Python 3.9+
- FFmpeg for audio processing
- Groq API key (free tier available)

# Clone repository
```
git clone https://github.com/yourusername/music-analysis-suite.git
cd music-analysis-suite
```
# Create virtual environment
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
# Install dependencies
```
pip install -r requirements.txt
```

# Set environment variables
```
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### Usage
# Launch Application:
```       
streamlit run app.py
```
# Workflow:

- Step 1: Upload audio file (MP3/WAV)
- Step 2: Enter song metadata and audio features
- Step 3: Chat with AI music expert

### Project Structure
music-analysis-suite/
├── app.py                 # Main application
├── models/
│   ├── song_preprocessor.pkl
│   └── viral_song_model.h5
├── utils/
│   ├── audio_processor.py
│   └── visualization.py
├── requirements.txt
└── README.md

### Thanks 
