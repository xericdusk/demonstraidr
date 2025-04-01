import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
import openai
import speech_recognition as sr
from scipy import signal
import base64
from io import BytesIO
import io
import re
import requests
import tempfile
import cv2  # For image processing
import gdown  # For downloading files from Google Drive

# Set page config
st.set_page_config(
    page_title="DemonstRAIDR - Tactical SIGINT Analyst",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure OpenAI API
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Store your API key in Streamlit secrets
except KeyError:
    st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY in Streamlit Cloud secrets.")
    st.stop()

# Load signal database
@st.cache_data
def load_signal_database():
    basic_db = {
        "commercial_drone": {"freq_ranges": [[2.4e9, 2.5e9], [5.7e9, 5.9e9]], "bandwidths": [1e6, 4e6, 8e6, 20e6], "threat_level": "medium"},
        "wifi": {"freq_ranges": [[2.4e9, 2.5e9], [5.1e9, 5.9e9]], "bandwidths": [20e6, 40e6, 80e6, 160e6], "threat_level": "low"},
        "pmr_radio": {"freq_ranges": [[440e6, 470e6]], "bandwidths": [12.5e3, 25e3], "threat_level": "low"},
        "cellular": {"freq_ranges": [[700e6, 900e6], [1.7e9, 2.1e9]], "bandwidths": [200e3, 1.4e6, 5e6, 10e6, 20e6], "threat_level": "low"},
        "military_tactical": {"freq_ranges": [[30e6, 88e6], [225e6, 400e6]], "bandwidths": [25e3, 5e6, 10e6], "threat_level": "high"},
        "fm_radio": {"freq_ranges": [[88e6, 108e6]], "bandwidths": [200e3], "threat_level": "low"},
        "amateur_radio": {"freq_ranges": [[1.8e6, 30e6], [50e6, 54e6], [144e6, 148e6]], "bandwidths": [10e3, 20e3], "threat_level": "low"},
        "tv_broadcast": {"freq_ranges": [[174e6, 216e6], [470e6, 694e6]], "bandwidths": [6e6], "threat_level": "low"},
        "gps": {"freq_ranges": [[1575.42e6, 1575.42e6]], "bandwidths": [2e6], "threat_level": "low"}
    }
    return basic_db

signal_db = load_signal_database()

def download_file_from_google_drive(file_url):
    """Download a file from Google Drive using its shareable link."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            temp_path = temp_file.name
            gdown.download(file_url, temp_path, quiet=False)
        
        with open(temp_path, 'rb') as f:
            raw_bytes = f.read()
        
        os.unlink(temp_path)
        
        if not raw_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            st.error(f"Downloaded file is not a valid PNG file. First 8 bytes: {raw_bytes[:8].hex()}")
            return None, None, None
        
        st.info(f"Downloaded file size: {len(raw_bytes)} bytes")
        st.info(f"First 8 bytes: {raw_bytes[:8].hex()}")
        
        file_name = "downloaded_file.png"
        file_content = io.BytesIO(raw_bytes)
        return file_content, file_name, raw_bytes
    except Exception as e:
        st.error(f"Error downloading file from Google Drive: {str(e)}")
        return None, None, None
    
def process_waterfall_image(image_data):
    try:
        file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not load image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        timestamp = "2025-03-31 18:46:49 to 18:51:06"
        freq_range = [10e6, 6000e6]
        power_range = [0.46, 20.69]

        height, width = gray.shape
        plot_top = int(height * 0.1)
        plot_bottom = int(height * 0.9)
        plot_left = int(width * 0.05)
        plot_right = int(width * 0.95)
        plot_img = gray[plot_top:plot_bottom, plot_left:plot_right]

        _, thresh = cv2.threshold(plot_img, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_signals = []
        plot_width = plot_right - plot_left
        freq_start, freq_end = freq_range

        def x_to_freq(x):
            norm_x = x / plot_width
            log_freq = np.log10(freq_start) + norm_x * (np.log10(freq_end) - np.log10(freq_start))
            return 10 ** log_freq

        def intensity_to_power(intensity):
            norm_intensity = intensity / 255.0
            return power_range[0] + norm_intensity * (power_range[1] - power_range[0])

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            freq = x_to_freq(cx)
            intensity = plot_img[cy, cx]
            power_dbm = intensity_to_power(intensity)
            x, y, w, h = cv2.boundingRect(contour)
            bw = (w / plot_width) * (freq_end - freq_start)

            signal_type = classify_signal(freq, bw)

            detected_signals.append({
                "frequency": float(freq),
                "power_dbm": float(power_dbm),
                "bandwidth": float(bw),
                "type": signal_type["type"],
                "threat_level": signal_type["threat_level"],
                "confidence": signal_type["confidence"]
            })

        detected_signals.sort(key=lambda x: x["frequency"])

        return {
            "timestamp": timestamp,
            "frequency_range": freq_range,
            "detected_signals": detected_signals,
            "is_waterfall": True,
            "image_data": image_data
        }
    except Exception as e:
        st.error(f"Error processing waterfall image: {str(e)}")
        return {}

def load_scan_file(uploaded_file):
    try:
        is_png = False
        if hasattr(uploaded_file, 'name') and uploaded_file.name.endswith('.png'):
            is_png = True
        else:
            uploaded_file.seek(0)
            header = uploaded_file.read(8)
            uploaded_file.seek(0)
            if header == b'\x89PNG\r\n\x1a\n':
                is_png = True

        if is_png:
            return process_waterfall_image(uploaded_file)
        elif hasattr(uploaded_file, 'name') and uploaded_file.name.endswith('.jsonl'):
            content = uploaded_file.read().decode('utf-8')
            for line in content.split('\n'):
                if line.strip():
                    return json.loads(line.strip())
        else:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
        
            if "timestamp_start" in data and "timestamp_end" in data:
                return {
                    "timestamp": f"{data['timestamp_start']} to {data['timestamp_end']}",
                    "frequency_range": data.get("frequency_range", [0, 0]),
                    "detected_signals": data.get("detected_signals", []),
                    "is_waterfall": True
                }
            else:
                return data
    except json.JSONDecodeError:
        if not is_png:
            uploaded_file.seek(0)
            header = uploaded_file.read(8)
            uploaded_file.seek(0)
            if header == b'\x89PNG\r\n\x1a\n':
                return process_waterfall_image(uploaded_file)
        st.error("File format not supported. Please upload a PNG, JSON, or JSONL file.")
        return {}
    except Exception as e:
        st.error(f"Error loading scan file: {str(e)}")
        return {}

def process_iq_data(iq_data, center_freq, sample_rate):
    freqs, psd = signal.welch(iq_data, fs=sample_rate, nperseg=1024, scaling='spectrum')
    freq_axis = freqs + (center_freq - sample_rate/2)
    peaks, _ = signal.find_peaks(psd, height=np.mean(psd)*3, distance=10)
    detected_signals = []
    for peak in peaks:
        peak_freq = freq_axis[peak]
        peak_power = 10 * np.log10(psd[peak])
        bw = estimate_bandwidth(psd, peak, sample_rate, len(psd))
        signal_type = classify_signal(peak_freq, bw)
        detected_signals.append({
            "frequency": float(peak_freq),
            "power_dbm": float(peak_power),
            "bandwidth": float(bw),
            "type": signal_type["type"],
            "threat_level": signal_type["threat_level"],
            "confidence": signal_type["confidence"]
        })
    return {"freq_axis": freq_axis.tolist(), "psd": psd.tolist(), "detected_signals": detected_signals}

def estimate_bandwidth(psd, peak_idx, sample_rate, n_points):
    peak_power = psd[peak_idx]
    threshold = peak_power / 2
    lower, upper = peak_idx, peak_idx
    while lower > 0 and psd[lower] > threshold:
        lower -= 1
    while upper < len(psd)-1 and psd[upper] > threshold:
        upper += 1
    return (upper - lower) * (sample_rate / n_points)

def classify_signal(frequency, bandwidth):
    best_match = {"type": "unknown", "threat_level": "unknown", "confidence": 0.0}
    for sig_type, sig_info in signal_db.items():
        in_range = any(freq_range[0] <= frequency <= freq_range[1] for freq_range in sig_info["freq_ranges"])
        if not in_range:
            continue
        bw_match = any(0.5 * bw <= bandwidth <= 2.0 * bw for bw in sig_info["bandwidths"])
        confidence = 0.7 if in_range else 0.0
        confidence += 0.3 if bw_match else 0.0
        if confidence > best_match["confidence"]:
            best_match = {"type": sig_type, "threat_level": sig_info["threat_level"], "confidence": confidence}
    if best_match["confidence"] == 0.0:
        if frequency < 30e6:
            best_match = {"type": "hf_signal", "threat_level": "low", "confidence": 0.3}
        elif frequency < 300e6:
            best_match = {"type": "vhf_signal", "threat_level": "low", "confidence": 0.3}
        elif frequency < 1e9:
            best_match = {"type": "uhf_signal", "threat_level": "low", "confidence": 0.3}
        else:
            best_match = {"type": "microwave_signal", "threat_level": "low", "confidence": 0.3}
    return best_match

def text_to_speech(text, voice="onyx"):
    try:
        nato_numbers = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'niner'
        }
        
        lines = text.split('\n')
        processed_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_text += "\n"
                continue
            
            if line.startswith("FREQ:"):
                freq_str = line.split("FREQ:")[1].split("MHz")[0].strip()
                freq_phonetic = ""
                for char in freq_str:
                    if char in nato_numbers:
                        freq_phonetic += nato_numbers[char] + " "
                    elif char == '.':
                        freq_phonetic += "point "
                freq_phonetic = freq_phonetic.strip()
                processed_text += f"[SPEED:0.8] {freq_phonetic} megahertz [SPEED:0.95]\n"
            else:
                processed_text += f"{line}\n"
        
        processed_text = processed_text.strip()
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
        
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": processed_text,
            "voice": voice,
            "response_format": "mp3",
            "speed": 0.95
        }
        
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=data,
            stream=True
        )
        
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                for data_chunk in response.iter_content(chunk_size=1024):
                    if data_chunk:
                        f.write(data_chunk)
            
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            os.unlink(temp_path)
            
            audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            st.success("Voice transmission complete")
            return True
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return False

def speech_to_text():
    st.warning("Speech-to-text is not supported in the cloud environment due to microphone access limitations.")
    return ""

def plot_spectrum(scan_data):
    if scan_data.get("is_waterfall", False):
        fig, ax = plt.subplots(figsize=(10, 5))
        signals = scan_data.get("detected_signals", [])
        if not signals:
            st.warning("No signals to plot in waterfall data.")
            return None
        
        freqs = [signal["frequency"] / 1e6 for signal in signals]
        powers = [signal.get("power_dbm", -100) for signal in signals]
        ax.scatter(freqs, powers, c='red', label='Detected Signals')
        
        for signal in signals:
            freq = signal["frequency"] / 1e6
            power = signal.get("power_dbm", -100)
            label = signal.get("type", "unknown")
            color = 'green' if signal.get("threat_level") == "low" else 'orange' if signal.get("threat_level") == "medium" else 'red'
            ax.plot(freq, power, 'o', markersize=8, color=color)
            ax.annotate(label, (freq, power), xytext=(0, 10), textcoords='offset points', ha='center')
        
        freq_range = scan_data.get("frequency_range", [0, 0])
        ax.set_xlim(freq_range[0] / 1e6, freq_range[1] / 1e6)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (dBm)')
        ax.set_title(f'RF Spectrum Waterfall: {scan_data.get("timestamp", "Unknown")}')
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        freq_axis = np.array(scan_data.get("freq_axis", []))
        psd = np.array(scan_data.get("psd", []))
        freq_mhz = freq_axis / 1e6
        ax.plot(freq_mhz, 10 * np.log10(psd), 'b-', linewidth=1)
        signals = scan_data.get("detected_signals", [])
        for signal in signals:
            freq = signal.get("frequency", 0) / 1e6
            power = signal.get("power_dbm", -100)
            label = signal.get("type", "unknown")
            color = 'green' if signal.get("threat_level") == "low" else 'orange' if signal.get("threat_level") == "medium" else 'red'
            ax.plot(freq, power, 'o', markersize=8, color=color)
            ax.annotate(label, (freq, power), xytext=(0, 10), textcoords='offset points', ha='center')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (dBm)')
        center_freq = scan_data.get("center_freq", 0) / 1e6
        ax.set_title(f'RF Spectrum at {center_freq} MHz')
        ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def get_available_scans(uploaded_files):
    scans = []
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        scan_data = load_scan_file(file)
        if scan_data:
            scans.append({
                "id": i,
                "timestamp": scan_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "center_freq": scan_data.get("frequency_range", [0, 0])[0] / 1e6 if scan_data.get("is_waterfall") else scan_data.get("center_freq", 0) / 1e6,
                "file_type": "png" if file.name.endswith('.png') else "json",
                "scan_data": scan_data,
                "file_content": file
            })
    return scans

def prepare_llm_context(selected_scan_data):
    """Prepare context for the LLM based on selected scan data."""
    context = "RAIDR Tactical SIGINT Analysis:\n"
    for scan in selected_scan_data:
        scan_data = scan["scan_data"]
        context += f"Scan ID: {scan['id']}\n"
        context += f"Timestamp: {scan['timestamp']}\n"
        context += f"File Type: {scan['file_type']}\n"
        signals = scan_data.get("detected_signals", [])
        if signals:
            context += "Detected Signals:\n"
            for signal in signals:
                context += f"- Frequency: {signal.get('frequency', 0) / 1e6:.2f} MHz, "
                context += f"Power: {signal.get('power_dbm', 0):.2f} dBm, "
                context += f"Bandwidth: {signal.get('bandwidth', 0) / 1e3:.2f} kHz, "
                context += f"Type: {signal.get('type', 'unknown')}, "
                context += f"Threat Level: {signal.get('threat_level', 'unknown')}, "
                context += f"Confidence: {signal.get('confidence', 0):.2f}\n"
        else:
            context += "No signals detected.\n"
        context += "\n"
    return context.strip()

def query_chatgpt(query, context):
    """Query OpenAI's ChatGPT with the given context and user query using the updated API."""
    try:
        client = openai.OpenAI(api_key=openai.api_key)  # Initialize the client
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tactical SIGINT analyst assistant. Provide concise, accurate responses based on the provided RF scan data."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error querying ChatGPT: {str(e)}")
        return "Unable to process query due to an error."

def main():
    st.markdown("""
    <style>
    .main {background-color: #1E1E1E; color: #DCDCDC;}
    .st-bw {background-color: #2D2D2D;}
    .stTextInput > div > div > input {background-color: #3D3D3D; color: #DCDCDC;}
    .tactical-header {color: #00FF00; text-align: center; font-family: 'Courier New', monospace; margin-bottom: 20px;}
    .tactical-text {font-family: 'Courier New', monospace; background-color: #2D2D2D; padding: 10px; border-radius: 5px; border-left: 3px solid #00FF00; white-space: pre-wrap;}
    .threat-high {color: #FF5555; font-weight: bold;}
    .threat-medium {color: #FFAA00; font-weight: bold;}
    .threat-low {color: #55FF55;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="tactical-header">DemonstRAIDR: Tactical SIGINT Analyst</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Scan Controls")
        
        # Input for Google Drive shareable link
        st.subheader("Load Scan from Google Drive Link")
        file_url = st.text_input("Enter Google Drive shareable link to a PNG file:", key="file_url")
        
        # File uploader for local PNG files
        st.subheader("Upload Local Scan File")
        uploaded_file = st.file_uploader("Choose a PNG file", type=["png"], key="file_uploader")
        
        uploaded_files = []
        
        # Handle Google Drive link
        if file_url:
            with st.spinner("Downloading file from Google Drive..."):
                file_content, file_name, raw_bytes = download_file_from_google_drive(file_url)
                if file_content and file_name:
                    uploaded_file_from_drive = type('UploadedFile', (), {
                        'read': lambda self: file_content.read(),
                        'seek': lambda self, pos: file_content.seek(pos),
                        'name': file_name
                    })()
                    file_content.seek(0)
                    uploaded_file_from_drive.seek(0)
                    uploaded_files.append(uploaded_file_from_drive)
                    st.success(f"Successfully downloaded file: {file_name}")
        
        # Handle local file upload
        if uploaded_file is not None:
            uploaded_files.append(uploaded_file)
            st.success(f"Successfully uploaded file: {uploaded_file.name}")
        
        if not uploaded_files:
            st.info("No files uploaded or downloaded. Please upload a PNG file or provide a valid Google Drive shareable link.")
        
        if uploaded_files:
            scans = get_available_scans(uploaded_files)
            if not scans:
                st.info("No valid scans loaded. Please ensure the file is a valid PNG.")
            else:
                scan_options = {f"{scan['id']} - {scan['timestamp'][:16]} ({scan['center_freq']} MHz) [{scan['file_type']}]": scan for scan in scans}
                selected_scans = st.multiselect("Select scans for analysis", options=list(scan_options.keys()), default=list(scan_options.keys()))
                selected_scan_data = [scan_options[scan] for scan in selected_scans]
    
    tab1, tab2 = st.tabs(["Spectrum Analyzer", "Tactical SIGINT"])
    
    with tab1:
        st.header("RF Spectrum Analysis")
        if 'selected_scan_data' in locals() and selected_scan_data:
            scan = selected_scan_data[0]
            scan_data = scan["scan_data"]
            if scan["file_type"] == "png":
                st.image(scan["file_content"], use_container_width=True, caption="Original Waterfall Plot")
            else:
                img_str = plot_spectrum(scan_data)
                if img_str:
                    st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
            st.subheader("Detected Signals")
            signals = scan_data.get("detected_signals", [])
            if signals:
                signals_df = pd.DataFrame([
                    {
                        "Frequency (MHz)": signal.get("frequency", 0) / 1e6,
                        "Power (dBm)": signal.get("power_dbm", 0),
                        "Bandwidth (kHz)": signal.get("bandwidth", 0) / 1e3 if "bandwidth" in signal else "N/A",
                        "Type": signal.get("type", "unknown"),
                        "Threat Level": signal.get("threat_level", "unknown"),
                        "Confidence": f"{signal.get('confidence', 0):.1f}"
                    } for signal in signals
                ])
                def highlight_threats(val):
                    return 'color: red; font-weight: bold' if val == "high" else 'color: orange; font-weight: bold' if val == "medium" else 'color: green' if val == "low" else ''
                styled_df = signals_df.style.applymap(highlight_threats, subset=["Threat Level"])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No signals detected in this scan.")
        else:
            st.info("No scans available for analysis. Please upload a PNG file or provide a valid Google Drive shareable link.")
    
    with tab2:
        st.header("RAIDR Tactical SIGINT Interface")
        if 'selected_scan_data' in locals() and selected_scan_data:
            context = prepare_llm_context(selected_scan_data)
            query = st.text_input("Enter tactical query:", key="text_query")
            
            voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            selected_voice = st.selectbox("Select TTS Voice", voice_options, index=3)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Send Query"):
                    if query:
                        with st.spinner("RAIDR processing..."):
                            response = query_chatgpt(query, context)
                            st.session_state.last_response = response
                            st.session_state.show_response = True
            with col2:
                if st.button("Voice Query"):
                    with st.spinner("Listening for voice query..."):
                        voice_query = speech_to_text()
                        if voice_query:
                            st.success(f"Voice query detected: \"{voice_query}\"")
                            with st.spinner("RAIDR processing..."):
                                response = query_chatgpt(voice_query, context)
                                st.session_state.last_response = response
                                st.session_state.show_response = True
            
            if st.session_state.get('show_response', False):
                st.subheader("RAIDR Response:")
                st.markdown(f'<div class="tactical-text">{st.session_state.last_response}</div>', unsafe_allow_html=True)
                if st.button("Speak Response"):
                    text_to_speech(st.session_state.last_response, selected_voice)
            with st.expander("Raw Spectrum Data Context"):
                st.text(context)
        else:
            st.info("No scans available for RAIDR analysis. Please upload a PNG file or provide a valid Google Drive shareable link.")

if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'show_response' not in st.session_state:
    st.session_state.show_response = False

if __name__ == "__main__":
    main()