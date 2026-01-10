import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI-NIDS Student Project", layout="wide")

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**: This system uses **Random Forest** to detect Network attacks and **Groq AI** to explain the packets.
""")

# --- CONFIGURATION ---
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input("Groq API Key (starts with gsk_)", type="password")
st.sidebar.caption("[Get a free key here](https://console.groq.com/keys)")

st.sidebar.header("2. Model Training")

@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, nrows=15000)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        return None

def train_model(df):
    features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                'Total Length of Fwd Packets', 'Fwd Packet Length Max', 
                'Flow IAT Mean', 'Flow IAT Std', 'Flow Packets/s']
    target = 'Label'
    
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        return None, 0, [], None, None

    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    score = accuracy_score(y_test, clf.predict(X_test))
    return clf, score, features, X_test, y_test

# --- APP LOGIC ---
df = load_data(DATA_FILE)

if df is None:
    st.error(f"Error: File '{DATA_FILE}' not found. Please upload it to the Files tab.")
    st.stop()

st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

if st.sidebar.button("Train Model Now"):
    with st.spinner("Training model..."):
        clf, accuracy, feature_names, X_test, y_test = train_model(df)
        if clf:
            st.session_state['model'] = clf
            st.session_state['features'] = feature_names
            st.session_state['X_test'] = X_test 
            st.session_state['y_test'] = y_test
            st.sidebar.success(f"Training Complete! Accuracy: {accuracy:.2%}")

st.header("3. Threat Analysis Dashboard")

if 'model' in st.session_state:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulation")
        st.info("Pick a random packet from the test data to simulate live traffic.")
        
        if st.button("🎲 Capture Random Packet"):
            random_idx = np.random.randint(0, len(st.session_state['X_test']))
            packet_data = st.session_state['X_test'].iloc[random_idx]
            actual_label = st.session_state['y_test'].iloc[random_idx]
            
            st.session_state['current_packet'] = packet_data
            st.session_state['actual_label'] = actual_label
            
    if 'current_packet' in st.session_state:
        packet = st.session_state['current_packet']
        
        with col1:
            st.write("**Packet Header Info:**")
            st.dataframe(packet, use_container_width=True)

        with col2:
            st.subheader("AI Detection Result")
            prediction = st.session_state['model'].predict([packet])[0]
            
            if prediction == "BENIGN":
                st.success(f" STATUS: **SAFE (BENIGN)**")
            else:
                st.error(f"🚨 STATUS: **ATTACK DETECTED ({prediction})**")
            
            st.caption(f"Ground Truth Label: {st.session_state['actual_label']}")

            st.markdown("---")
            st.subheader(" Ask AI Analyst (Groq)")
            
            if st.button("Generate Explanation"):
                if not groq_api_key:
                    st.warning(" Please enter your Groq API Key in the sidebar first.")
                else:
                    try:
                        client = Groq(api_key=groq_api_key)
                        
                        prompt = f"""
                        You are a cybersecurity analyst. 
                        A network packet was detected as: {prediction}.
                        
                        Packet Technical Details:
                        {packet.to_string()}
                        
                        Please explain:
                        1. Why these specific values (like Flow Duration or Packet Length) might indicate {prediction}.
                        2. If it is BENIGN, explain why it looks normal.
                        3. Keep the answer short and simple for a student.
                        """

                        with st.spinner("Groq is analyzing the packet..."):
                            completion = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",  # <--- UPDATED MODEL NAME
                                messages=[
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.6,
                            )
                            st.info(completion.choices[0].message.content)
                            
                    except Exception as e:
                        st.error(f"API Error: {e}")
else:
    st.info(" Waiting for model training. Click **'Train Model Now'** in the sidebar.")