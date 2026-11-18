import streamlit as st
import pandas as pd
import joblib
import json
import os

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Bearing Fault Diagnosis",
    page_icon="⚙️",
    layout="wide"
)

MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"

# ==========================================
# LOAD MODEL DAN METADATA
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return model, metadata

model, metadata = load_model()

# ==========================================
# HEADER
# ==========================================

st.title("⚙️ Bearing Fault Diagnosis")
st.subheader("Machine Learning-Based Bearing Condition Classification")

st.write(
    "Aplikasi untuk mengidentifikasi kondisi bearing "
    "berdasarkan parameter pengukuran getaran mesin."
)

st.divider()

# ==========================================
# INPUT DATA
# ==========================================

st.header("Input Data Pengukuran")

with st.form("bearing_form"):

    posisi_sensor = st.selectbox(
        "Posisi Sensor",
        ["drive_end", "fan_end"]
    )

    col1, col2 = st.columns(2)

    with col1:
        kecepatan_rpm = st.number_input(
            "Kecepatan (RPM)", min_value=0.0, value=2304.0
        )
        beban_kN = st.number_input(
            "Beban (kN)", min_value=0.0, value=2.3
        )
        suhu_housing_C = st.number_input(
            "Suhu Housing (°C)", value=50.8
        )
        rms_g = st.number_input(
            "RMS (g)", min_value=0.0, value=0.4946
        )
        puncak_g = st.number_input(
            "Puncak (g)", min_value=0.0, value=1.4368
        )
        kurtosis = st.number_input(
            "Kurtosis", value=2.209
        )

    with col2:
        crest_factor = st.number_input(
            "Crest Factor", min_value=0.0, value=2.905
        )
        energi_bpfo = st.number_input(
            "Energi BPFO", min_value=0.0, value=0.0042
        )
        energi_bpfi = st.number_input(
            "Energi BPFI", min_value=0.0, value=0.0834
        )
        energi_bsf = st.number_input(
            "Energi BSF", min_value=0.0, value=0.1533
        )
        frekuensi_dominan_hz = st.number_input(
            "Frekuensi Dominan (Hz)",
            min_value=0.0,
            value=38.7
        )

    submitted = st.form_submit_button(
        "Prediksi Kondisi Bearing",
        use_container_width=True
    )

# ==========================================
# PREDIKSI
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

    # Pastikan urutan fitur sama dengan saat training
    input_data = input_data[metadata["features"]]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.divider()
    st.header("Hasil Prediksi")

    st.success(
        f"Kondisi bearing terdeteksi: **{prediction.upper()}**"
    )

    # Probabilitas
    probability_df = pd.DataFrame({
        "Kondisi": model.classes_,
        "Probabilitas": probabilities
    }).sort_values(
        "Probabilitas",
        ascending=False
    )

    st.subheader("Probabilitas Prediksi")

    probability_df["Persentase"] = (
        probability_df["Probabilitas"] * 100
    )

    st.bar_chart(
        probability_df.set_index("Kondisi")["Persentase"]
    )

    # Rekomendasi
    recommendations = {
        "normal": (
            "Bearing dalam kondisi normal. "
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

    st.subheader("Rekomendasi")
    st.info(recommendations[prediction])

    st.caption(
        "Hasil merupakan prediksi model machine learning "
        "dan perlu dikonfirmasi melalui inspeksi teknis."
    )

# ==========================================
# INFORMASI MODEL
# ==========================================

with st.expander("Informasi Model"):
    st.write("Nama model:", metadata["model_name"])
    st.write("Target:", metadata["target"])
    st.write("Validation Macro F1:", metadata["validation_macro_f1"])
    st.write("Test Macro F1:", metadata["test_macro_f1"])
    st.write("Test Accuracy:", metadata["test_accuracy"])
