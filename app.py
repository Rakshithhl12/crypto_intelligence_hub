# app.py
import streamlit as st
from crypto_dashboard import dashboard  # your dashboard code from crypto_dashboard.py

# ----------------- Page Config -----------------
st.set_page_config(page_title="Crypto Intelligence Hub", page_icon="💎", layout="wide")

# ----------------- Run Dashboard -----------------
dashboard()  # No authentication needed
