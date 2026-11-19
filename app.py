
import streamlit as st
import pandas as pd
import joblib
import json
import os

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="BearingGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5edf7;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}
header {visibility: hidden;}

.hero {
    background: linear-gradient(120deg, #14243c, #193c5b);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #294966;
    margin-bottom: 25px;
}
.hero-label {
    color: #65c7f7;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}
.hero h1 {
    color: white;
    font-size: 36px;
    margin: 8px 0;
}
.hero p {
    color: #c1d3e5;
    margin-bottom: 0;
}
.section-title {
    color: #f1f5f9;
    font-size: 21px;
    font-weight: 700;
    margin-top: 18px;
}
.metric-card {
    background: #142238;
    border: 1px solid #263c57;
    padding: 18px;
    border-radius: 14px;
    min-height: 105px;
}
.metric-label {
    color: #9eb2c9;
    font-size: 13px;
}
.metric-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 700;
    margin-top: 8px;
}
.result-card {
    background: #142238;
    border: 1px solid #34516f;
    border-radius: 16px;
    padding: 25px;
    margin-top: 15px;
}
.result-label {
    color: #9eb2c9;
    font-size: 13px;
}
.result-value {
    color: #65c7f7;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}
div[data-testid="stForm"] {
    background: #111e31;
    border: 1px solid #263c57;
    border-radius: 16px;
    padding: 22px;
}
.stButton button, 
.stFormSubmitButton button {
    background: #1687c9;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}
.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #0874b5;
    color: white;
}
div[data-testid="stExpander"] {
    background: #111e31;
    border: 1px solid #263c57;
    border-radius: 12px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return model, metadata

if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    st.error("File model atau metadata tidak ditemukan. Periksa folder models/.")
    st.stop()

model, metadata = load_model()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-label">INDUSTRIAL AI • PREDICTIVE MAINTENANCE</div>
    <h1>⚙️ BearingGuard AI</h1>
    <p>Intelligent Bearing Fault Diagnosis System</p>
    <p>Platform klasifikasi kondisi bearing berdasarkan parameter pengukuran getaran mesin.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MODEL SUMMARY
# ==========================================

st.markdown('<div class="section-title">System Overview</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL ALGORITHM</div>
        <div class="metric-value">Extra Trees</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TEST MACRO F1</div>
        <div class="metric-value">{metadata['test_macro_f1']:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TEST ACCURACY</div>
        <div class="metric-value">{metadata['test_accuracy']:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# INPUT FORM
# ==========================================

st.markdown('<div class="section-title">📊 Input Measurement Data</div>',
            unsafe_allow_html=True)

st.caption("Masukkan data hasil pengukuran getaran bearing untuk melakukan klasifikasi.")

with st.form("bearing_form"):

    posisi_sensor = st.selectbox(
        "Posisi Sensor",
        ["drive_end", "fan_end"]
    )

    st.markdown("#### Operating Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        kecepatan_rpm = st.number_input(
            "Kecepatan (RPM)", min_value=0.0, value=2304.0
        )
        beban_kN = st.number_input(
            "Beban (kN)", min_value=0.0, value=2.3
        )

    with col2:
        suhu_housing_C = st.number_input(
            "Suhu Housing (°C)", value=50.8
        )
        frekuensi_dominan_hz = st.number_input(
            "Frekuensi Dominan (Hz)", min_value=0.0, value=38.7
        )

    with col3:
        rms_g = st.number_input(
            "RMS (g)", min_value=0.0, value=0.4946
        )
        puncak_g = st.number_input(
            "Puncak (g)", min_value=0.0, value=1.4368
        )

    st.markdown("#### Vibration Characteristics")
    col4, col5, col6 = st.columns(3)

    with col4:
        kurtosis = st.number_input(
            "Kurtosis", value=2.209
        )
        crest_factor = st.number_input(
            "Crest Factor", min_value=0.0, value=2.905
        )

    with col5:
        energi_bpfo = st.number_input(
            "Energi BPFO", min_value=0.0, value=0.0042
        )
        energi_bpfi = st.number_input(
            "Energi BPFI", min_value=0.0, value=0.0834
        )

    with col6:
        energi_bsf = st.number_input(
            "Energi BSF", min_value=0.0, value=0.1533
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 ANALYZE BEARING CONDITION",
        use_container_width=True
    )

# ==========================================
# PREDICTION
# ==========================================

if submitted:

    sensor_mapping = {
        "drive_end": 0,
        "fan_end": 1
    }

    input_data = pd.DataFrame([{
        "posisi_sensor": sensor_mapping[posisi_sensor],
        "kecepatan_rpm": kecepatan_rpm,
        "beban_kN": beban_kN,
        "suhu_housing_C": suhu_housing_C,
        "rms_g": rms_g,
        "puncak_g": puncak_g,
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "energi_bpfo": energi_bpfo,
        "energi_bpfi": energi_bpfi,
        "energi_bsf": energi_bsf,
        "frekuensi_dominan_hz": frekuensi_dominan_hz
    }])

    input_data = input_data[metadata["features"]]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    recommendations = {
        "normal": (
            "Bearing terindikasi dalam kondisi normal. "
            "Lanjutkan pemantauan dan perawatan berkala."
        ),
        "outer_race": (
            "Terindikasi kerusakan outer race. "
            "Lakukan inspeksi dan pertimbangkan penggantian bearing."
        ),
        "inner_race": (
            "Terindikasi kerusakan inner race. "
            "Jadwalkan pemeriksaan dan siapkan komponen pengganti."
        ),
        "bola": (
            "Terindikasi kerusakan elemen bola. "
            "Periksa kondisi bearing dan siapkan komponen pengganti."
        )
    }

    result_colors = {
        "normal": "#22c55e",
        "outer_race": "#f97316",
        "inner_race": "#ef4444",
        "bola": "#eab308"
    }

    result_color = result_colors.get(prediction, "#65c7f7")

    st.markdown('<div class="section-title">🧠 Diagnosis Result</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">PREDICTED BEARING CONDITION</div>
        <div style="color:{result_color}; font-size:32px; font-weight:800; margin-top:8px;">
            {prediction.upper()}
        </div>
        <div style="color:#9eb2c9; margin-top:8px;">
            Posisi sensor: {posisi_sensor}
        </div>
    </div>
    """, unsafe_allow_html=True)

    probability_df = pd.DataFrame({
        "Kondisi": model.classes_,
        "Probabilitas": probabilities * 100
    }).sort_values("Probabilitas", ascending=False)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Prediction Confidence")
        st.bar_chart(
            probability_df.set_index("Kondisi")["Probabilitas"],
            horizontal=True
        )

        st.dataframe(
            probability_df.style.format({
                "Probabilitas": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    with col_b:
        st.markdown("#### Maintenance Recommendation")
        st.info(recommendations[prediction])

        st.markdown("#### Inspection Reminder")
        if prediction == "normal":
            st.success("Pertahankan jadwal pemantauan rutin.")
        else:
            st.warning(
                "Lakukan inspeksi teknis lanjutan sebelum menentukan tindakan perbaikan."
            )

    st.caption(
        "Hasil merupakan prediksi model machine learning dan harus "
        "dikonfirmasi melalui inspeksi teknis."
    )

# ==========================================
# MODEL INFORMATION
# ==========================================

with st.expander("⚙️ Model Information & Evaluation"):
    st.write("**Model:**", metadata["model_name"])
    st.write("**Algorithm:**", metadata["model_type"])
    st.write("**Target:**", metadata["target"])
    st.write("**Validation Macro F1:**",
             f"{metadata['validation_macro_f1']:.4f}")
    st.write("**Test Macro F1:**",
             f"{metadata['test_macro_f1']:.4f}")
    st.write("**Test Accuracy:**",
             f"{metadata['test_accuracy']:.4f}")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8296ad; font-size:12px;">
    BearingGuard AI • Machine Learning Project • Bearing Fault Diagnosis
    <br>Decision Support System for Predictive Maintenance
</div>
""", unsafe_allow_html=True)
