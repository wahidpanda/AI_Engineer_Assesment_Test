import streamlit as st
import json
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from collections import defaultdict
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import joblib
from tensorflow.keras.models import load_model

# Initialize Groq LLM
groq_api_key = "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"
llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192", api_key=groq_api_key)

# File paths - UPDATE THESE TO YOUR ACTUAL FILE PATHS
MODEL_FILES = {
    'preprocessor': 'song_preprocessor.pkl',
    'model': 'viral_song_model.h5',
    'class_names': 'class_names.json'
}

# Load virality prediction models with proper error handling
@st.cache_resource
def load_virality_models():
    try:
        # Verify all files exist first
        missing_files = [name for name, path in MODEL_FILES.items() if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
        
        preprocessor = joblib.load(MODEL_FILES['preprocessor'])
        model = load_model(MODEL_FILES['model'])
        
        with open(MODEL_FILES['class_names']) as f:
            class_info = json.load(f)
        
        return {
            'preprocessor': preprocessor,
            'model': model,
            'classes': class_info['classes'],
            'error': None
        }
    except Exception as e:
        return {
            'preprocessor': None,
            'model': None,
            'classes': None,
            'error': str(e)
        }

# Instrument Detector Class
class InstrumentDetector:
    def __init__(self):
        try:
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            self.class_names = self._get_class_names()
            self.instrument_mapping = {
                'guitar': ['guitar', 'acoustic guitar', 'electric guitar'],
                'piano': ['piano', 'keyboard'],
                'drums': ['drum', 'drum kit', 'snare drum'],
                'bass': ['bass', 'bass guitar'],
                'violin': ['violin', 'fiddle'],
                'vocals': ['singing', 'voice'],
                'synthesizer': ['synthesizer'],
                'brass': ['trumpet', 'trombone'],
                'woodwind': ['flute', 'clarinet']
            }
            self.initialized = True
        except Exception as e:
            self.initialized = False
            st.error(f"Failed to initialize InstrumentDetector: {str(e)}")

    def _get_class_names(self):
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path) as f:
            return [line.strip().split(',')[2] for line in f]

    def _preprocess_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        audio = audio / np.max(np.abs(audio))
        return audio

    def _analyze_audio(self, audio):
        scores, _, _ = self.model(audio)
        mean_scores = np.mean(scores, axis=0)
        return mean_scores

    def _map_to_instruments(self, scores):
        instrument_scores = defaultdict(float)
        total_score = 0
        for class_idx, score in enumerate(scores):
            class_name = self.class_names[class_idx].lower()
            for instrument, keywords in self.instrument_mapping.items():
                if any(keyword in class_name for keyword in keywords):
                    instrument_scores[instrument] += float(score)
                    total_score += float(score)
                    break
        if total_score > 0:
            return {inst: round((score/total_score)*100, 2) for inst, score in instrument_scores.items()}
        return {}

    def detect_instruments(self, file_path):
        if not self.initialized:
            return {"status": "error", "message": "Instrument detector not initialized"}
        try:
            if not os.path.exists(file_path):
                return {"status": "error", "message": "File not found"}
            audio = self._preprocess_audio(file_path)
            scores = self._analyze_audio(audio)
            results = self._map_to_instruments(scores)
            return {
                "status": "success",
                "analysis": {k: float(v) for k, v in results.items()}
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Streamlit UI
def main():
    st.set_page_config(page_title="Music Analysis Chat", layout="wide")
    st.title("🎵 Music Analysis Chat")
    
    # Initialize session state
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Load models with error display
    models = load_virality_models()
    instrument_detector = InstrumentDetector()
    
    if models['error']:
        st.error(f"Model loading error: {models['error']}")
        st.info("Please ensure these files exist in your working directory:")
        st.code("\n".join(MODEL_FILES.values()))
        return
    
    # Step 1: Upload and instrument detection
    with st.expander("Step 1: Upload Song & Detect Instruments", expanded=True):
        audio_file = st.file_uploader("Upload your song (MP3/WAV)", type=["wav", "mp3"])
        
        if audio_file and st.button("Analyze Instruments"):
            with st.spinner("Detecting instruments..."):
                with open("temp_audio", "wb") as f:
                    f.write(audio_file.getbuffer())
                result = instrument_detector.detect_instruments("temp_audio")
                
                if result["status"] == "success":
                    st.session_state.analysis_data = {
                        "instruments": result["analysis"],
                        "virality": None
                    }
                    st.success("Instrument detection complete!")
                    st.json(result["analysis"])
                else:
                    st.error(f"Error: {result['message']}")
    
    # Step 2: Virality prediction
    if st.session_state.analysis_data and not st.session_state.analysis_data["virality"]:
        with st.expander("Step 2: Predict Virality", expanded=True):
            st.write("Provide additional song features:")
            
            col1, col2 = st.columns(2)
            with col1:
                danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
                energy = st.slider("Energy", 0.0, 1.0, 0.5)
                key = st.slider("Key (0-11)", 0, 11, 5)
                loudness = st.slider("Loudness (dB)", -60, 0, -10)
                speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
                acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
            
            with col2:
                instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
                liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
                valence = st.slider("Valence", 0.0, 1.0, 0.5)
                tempo = st.slider("Tempo (BPM)", 50, 200, 120)
                time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)
                genre = st.selectbox("Genre", ["pop", "rock", "electronic", "hiphop", "classical"])
                year = st.slider("Year", 1900, 2025, 2023)
                mode = st.radio("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
            
            if st.button("Predict Virality"):
                features = {
                    'danceability': danceability, 'energy': energy, 'key': key,
                    'loudness': loudness, 'speechiness': speechiness, 'acousticness': acousticness,
                    'instrumentalness': instrumentalness, 'liveness': liveness, 'valence': valence,
                    'tempo': tempo, 'time_signature': time_signature, 'genre': genre,
                    'year': year, 'mode': mode
                }
                
                with st.spinner("Predicting virality..."):
                    try:
                        # Create input DataFrame
                        input_df = pd.DataFrame([features])
                        input_df['energy_danceability'] = input_df['energy'] * input_df['danceability']
                        input_df['mood_score'] = (input_df['valence'] + input_df['energy']) / 2
                        
                        # Add required dummy columns
                        dummy_cols = {
                            'artist_name': 'unknown', 'track_name': 'unknown',
                            'Unnamed: 0': 0, 'track_id': 'unknown',
                            'duration_ms': 180000, 'duration_min': 3.0
                        }
                        for col, val in dummy_cols.items():
                            input_df[col] = val
                        
                        # Transform and predict
                        processed = models['preprocessor'].transform(input_df)
                        probabilities = models['model'].predict(processed)[0]
                        
                        prediction = {class_name: float(prob) for class_name, prob in zip(models['classes'], probabilities)}
                        st.session_state.analysis_data["virality"] = prediction
                        st.success("Virality prediction complete!")
                        st.json(prediction)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
    
    # Step 3: Chat with LLM
    if st.session_state.analysis_data and st.session_state.analysis_data["virality"]:
        st.divider()
        st.header("Step 3: Chat with Music Expert AI")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your song..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        template = """As a music industry expert, analyze this song:
                        
                        Instrument Analysis:
                        {instruments}
                        
                        Virality Prediction:
                        {virality}
                        
                        User Question: {question}
                        
                        Provide specific, actionable recommendations in this format:
                        
                        ### Analysis
                        - Key strengths
                        - Potential weaknesses
                        
                        ### Recommendations
                        - Musical improvements
                        - Target audiences
                        - Marketing strategies
                        - Similar successful tracks"""
                        
                        prompt_template = ChatPromptTemplate.from_template(template)
                        chain = prompt_template | llm | StrOutputParser()
                        
                        response = chain.invoke({
                            "instruments": json.dumps(st.session_state.analysis_data["instruments"], indent=2),
                            "virality": json.dumps(st.session_state.analysis_data["virality"], indent=2),
                            "question": prompt
                        })
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()