
import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BearingGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. THEME & CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background-color: #FFF8EB;
    color: #26365D;
    font-family: 'Trebuchet MS', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] {
    background-color: #FFF8EB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* TEXT */
h1, h2, h3, h4, p, label, span {
    color: #26365D;
}

h1 {
    font-weight: 900;
}

h2, h3 {
    font-weight: 800;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #26365D;
    border-right: 4px solid #FFD43B;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stRadio label {
    background-color: #354A78;
    padding: 12px 15px;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* HERO */
.hero {
    background: linear-gradient(120deg, #26365D, #405A91);
    border: 3px solid #FFD43B;
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 25px;
    box-shadow: 0 7px 18px rgba(38,54,93,0.15);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #FFFFFF !important;
    font-size: 15px;
}

.hero small {
    color: #FFD43B !important;
    font-weight: bold;
    letter-spacing: 2px;
}

/* CARDS */
.custom-card {
    background: #FFFFFF;
    border: 2px solid #E8D8BD;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 5px 15px rgba(38,54,93,0.06);
}

.metric-card {
    background: #FFFFFF;
    border: 2px solid #DCE4F2;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    min-height: 110px;
}

.metric-label {
    color: #526487 !important;
    font-size: 13px;
    font-weight: bold;
}

.metric-value {
    color: #26365D !important;
    font-size: 25px;
    font-weight: 900;
}

/* SECTION */
.section-title {
    color: #26365D !important;
    font-size: 24px;
    font-weight: 900;
    margin: 12px 0 18px 0;
}

/* INPUT */
.stNumberInput label,
.stSelectbox label {
    color: #26365D !important;
    font-weight: bold !important;
}

.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #26365D !important;
    border: 1px solid #AAB9D1 !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
}

/* BUTTON */
.stButton > button,
.stFormSubmitButton > button {
    background: #D9233F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 900 !important;
    padding: 12px !important;
    min-height: 45px;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #B81732 !important;
    transform: translateY(-2px);
}

/* RESULT BUBBLE */
.result-bubble {
    background: #FFFFFF;
    border: 3px solid #FFD43B;
    border-radius: 22px;
    padding: 25px;
    text-align: center;
    margin: 12px 0 20px 0;
    box-shadow: 0 7px 18px rgba(38,54,93,0.09);
}

.result-bubble .emoji {
    font-size: 48px;
}

.result-bubble .result-label {
    color: #526487 !important;
    font-size: 13px;
    font-weight: bold;
}

.result-bubble .result-value {
    font-size: 30px;
    font-weight: 900;
    margin: 8px 0;
}

/* INFO BOX */
.info-box {
    background: #FFF1C2;
    border-left: 6px solid #E7AD18;
    border-radius: 12px;
    padding: 15px;
    color: #59430B !important;
    margin-top: 12px;
}

/* RECOMMENDATION */
.recommendation {
    background: #EAF1FF;
    border-left: 6px solid #405A91;
    border-radius: 12px;
    padding: 17px;
    color: #26365D !important;
}

/* TABLE */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE4F2;
    border-radius: 12px;
}

/* MOBILE */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem 0.8rem;
    }

    .hero {
        padding: 18px;
    }

    .hero h1 {
        font-size: 25px;
    }

    .section-title {
        font-size: 20px;
    }

    .result-bubble .result-value {
        font-size: 24px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. LOAD MODEL AND METADATA
# =========================================================

MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "File models/model.joblib tidak ditemukan."
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "File models/metadata.json tidak ditemukan."
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata

try:
    model, metadata = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =========================================================
# 4. FEATURE CONFIGURATION
# =========================================================

FEATURE_COLUMNS = metadata["features"]

SENSOR_MAPPING = {
    "drive_end": 0,
    "fan_end": 1
}

CLASS_INFO = {
    "normal": {
        "emoji": "💚",
        "color": "#16834A",
        "title": "NORMAL",
        "recommendation": (
            "Bearing terdeteksi dalam kondisi normal. "
            "Lanjutkan pemantauan getaran dan perawatan berkala."
        )
    },
    "outer_race": {
        "emoji": "🔴",
        "color": "#D9233F",
        "title": "OUTER RACE",
        "recommendation": (
            "Terindikasi kerusakan pada outer race. "
            "Lakukan inspeksi dan pertimbangkan penggantian bearing."
        )
    },
    "inner_race": {
        "emoji": "🟠",
        "color": "#D97706",
        "title": "INNER RACE",
        "recommendation": (
            "Terindikasi kerusakan pada inner race. "
            "Jadwalkan pemeriksaan dan siapkan komponen pengganti."
        )
    },
    "bola": {
        "emoji": "🟣",
        "color": "#7851A9",
        "title": "BALL FAULT",
        "recommendation": (
            "Terindikasi kerusakan elemen bola. "
            "Periksa kondisi bearing dan siapkan komponen pengganti."
        )
    }
}

# =========================================================
# 5. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:20px 5px;">
            <div style="font-size:48px;">⚙️</div>
            <h2 style="color:white !important;">
                BearingGuard
            </h2>
            <p style="color:#FFD43B !important;">
                SMART BEARING DIAGNOSTIC SYSTEM
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🏠 MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔧 Diagnosis",
            "📋 Riwayat Prediksi",
            "📈 Informasi Model"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            background:#354A78;
            padding:17px;
            border-radius:15px;
            border:1px solid #6379A5;
        ">
            <h4 style="color:#FFD43B !important;">
                🌟 Shinchan Mode
            </h4>
            <p style="font-size:13px;color:white !important;">
                Keep your machines happy and healthy!
            </p>
            <div style="font-size:24px;">
                🧒🏻 🔧 ⚙️
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("BearingGuard AI | Machine Learning Project")

# =========================================================
# 7. HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <small>INDUSTRIAL AI • PREDICTIVE MAINTENANCE</small>
        <h1>⚙️ BearingGuard AI</h1>
        <p>
            Intelligent Bearing Fault Diagnosis System
        </p>
        <p>
            Platform klasifikasi kondisi bearing berdasarkan
            parameter pengukuran getaran mesin.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 8. HELPER: RESULT DISPLAY
# =========================================================

def show_result(result):

    prediction = result["prediction"]

    info = CLASS_INFO.get(
        prediction,
        {
            "emoji": "⚙️",
            "color": "#26365D",
            "title": prediction.upper(),
            "recommendation": "Lakukan pemeriksaan teknis."
        }
    )

    col_result, col_recommendation = st.columns(
        [1.15, 0.85]
    )

    with col_result:

        st.markdown(
            '<div class="section-title">✨ Diagnosis Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-bubble">
                <div class="emoji">{info["emoji"]}</div>
                <div class="result-label">
                    PREDICTED BEARING CONDITION
                </div>
                <div class="result-value"
                     style="color:{info["color"]};">
                    {info["title"]}
                </div>
                <div style="color:#526487;font-size:13px;">
                    Posisi sensor: {result["sensor"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_recommendation:

        st.markdown(
            '<div class="section-title">🛠️ Maintenance Recommendation</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="recommendation">
                <b>{info["emoji"]} Rekomendasi:</b>
                <p>{info["recommendation"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box">
                ⚠️ Hasil prediksi merupakan alat bantu analisis.
                Konfirmasikan melalui inspeksi teknis.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">📊 Prediction Confidence</div>',
        unsafe_allow_html=True
    )

    # Always normalize probabilities into a list of records
    records = result["probabilities"]

    if isinstance(records, dict):
        records = [
            {
                "Kondisi": key,
                "Probabilitas": float(value)
            }
            for key, value in records.items()
        ]

    probability_df = pd.DataFrame(records)

    if probability_df.empty:
        st.warning("Data probabilitas tidak tersedia.")
        return

    if "Kondisi" not in probability_df.columns:
        st.error("Kolom Kondisi tidak ditemukan.")
        return

    if "Probabilitas" not in probability_df.columns:
        st.error("Kolom Probabilitas tidak ditemukan.")
        return

    probability_df["Probabilitas"] = pd.to_numeric(
        probability_df["Probabilitas"],
        errors="coerce"
    ).fillna(0.0)

    probability_df["Persentase"] = (
        probability_df["Probabilitas"] * 100
    )

    probability_df = probability_df.sort_values(
        "Persentase",
        ascending=False
    )

    chart_col, table_col = st.columns([1.2, 0.8])

    with chart_col:

        chart_data = probability_df.set_index(
            "Kondisi"
        )["Persentase"]

        st.bar_chart(
            chart_data,
            horizontal=True
        )

    with table_col:

        display_df = probability_df[
            ["Kondisi", "Persentase"]
        ].copy()

        display_df["Persentase"] = (
            display_df["Persentase"].map(
                lambda x: f"{x:.2f}%"
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Probabilitas menunjukkan tingkat keyakinan model "
        "terhadap masing-masing kelas, bukan jaminan kondisi fisik."
    )

# =========================================================
# 9. DASHBOARD PAGE
# =========================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">📊 System Overview</div>',
        unsafe_allow_html=True
    )

    metric1, metric2, metric3 = st.columns(3)

    model_name = metadata.get(
        "model_name",
        "Extra Trees Classifier"
    )

    validation_score = metadata.get(
        "validation_macro_f1",
        0
    )

    test_accuracy = metadata.get(
        "test_accuracy",
        0
    )

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MODEL ALGORITHM</div>
                <div class="metric-value" style="font-size:19px;">
                    {model_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">VALIDATION MACRO F1</div>
                <div class="metric-value">
                    {validation_score * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TEST ACCURACY</div>
                <div class="metric-value">
                    {test_accuracy * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🧩 Bearing Condition Classes</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    dashboard_classes = [
        ("💚", "NORMAL", "Kondisi bearing normal."),
        ("🔴", "OUTER RACE", "Kerusakan lintasan luar."),
        ("🟠", "INNER RACE", "Kerusakan lintasan dalam."),
        ("🟣", "BALL FAULT", "Kerusakan elemen bola.")
    ]

    for col, item in zip(
        [c1, c2, c3, c4],
        dashboard_classes
    ):
        with col:
            st.markdown(
                f"""
                <div class="custom-card" style="text-align:center;">
                    <div style="font-size:35px;">{item[0]}</div>
                    <h4>{item[1]}</h4>
                    <p style="font-size:13px;">{item[2]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📝 Recent Predictions</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        recent = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            recent[
                ["timestamp", "sensor", "prediction"]
            ].tail(5),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Belum ada riwayat prediksi. "
            "Silakan buka menu Diagnosis untuk memulai."
        )

# =========================================================
# 10. DIAGNOSIS PAGE
# =========================================================

elif page == "🔧 Diagnosis":

    input_col, result_col = st.columns(
        [1.15, 0.85],
        gap="large"
    )

    with input_col:

        st.markdown(
            '<div class="section-title">🔧 Input Measurement</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Masukkan parameter pengukuran untuk "
            "menganalisis kondisi bearing."
        )

        with st.form("bearing_diagnosis_form"):

            posisi_sensor = st.selectbox(
                "📍 Posisi Sensor",
                ["drive_end", "fan_end"]
            )

            st.markdown("### ⚙️ Operating Parameters")

            c1, c2 = st.columns(2)

            with c1:

                kecepatan_rpm = st.number_input(
                    "Kecepatan (RPM)",
                    min_value=0.0,
                    value=2304.0,
                    step=1.0
                )

                beban_kN = st.number_input(
                    "Beban (kN)",
                    min_value=0.0,
                    value=2.30,
                    step=0.1
                )

                suhu_housing_C = st.number_input(
                    "Suhu Housing (°C)",
                    value=50.8,
                    step=0.1
                )

                rms_g = st.number_input(
                    "RMS (g)",
                    min_value=0.0,
                    value=0.4946,
                    step=0.01,
                    format="%.4f"
                )

                puncak_g = st.number_input(
                    "Puncak (g)",
                    min_value=0.0,
                    value=1.4368,
                    step=0.01,
                    format="%.4f"
                )

                kurtosis = st.number_input(
                    "Kurtosis",
                    value=2.209,
                    step=0.01
                )

            with c2:

                crest_factor = st.number_input(
                    "Crest Factor",
                    min_value=0.0,
                    value=2.905,
                    step=0.01
                )

                energi_bpfo = st.number_input(
                    "Energi BPFO",
                    min_value=0.0,
                    value=0.0042,
                    step=0.001,
                    format="%.5f"
                )

                energi_bpfi = st.number_input(
                    "Energi BPFI",
                    min_value=0.0,
                    value=0.0834,
                    step=0.001,
                    format="%.5f"
                )

                energi_bsf = st.number_input(
                    "Energi BSF",
                    min_value=0.0,
                    value=0.1533,
                    step=0.001,
                    format="%.5f"
                )

                frekuensi_dominan_hz = st.number_input(
                    "Frekuensi Dominan (Hz)",
                    min_value=0.0,
                    value=38.7,
                    step=0.1
                )

            st.markdown("---")

            submitted = st.form_submit_button(
                "🔍 ANALYZE BEARING CONDITION",
                use_container_width=True
            )

    # Run model after form submission
    if submitted:

        input_values = {
            "posisi_sensor": SENSOR_MAPPING[posisi_sensor],
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
        }

        try:

            input_df = pd.DataFrame(
                [input_values]
            )

            # Follow exact feature order from training metadata
            input_df = input_df[FEATURE_COLUMNS]

            prediction = str(
                model.predict(input_df)[0]
            )

            probabilities = model.predict_proba(
                input_df
            )[0]

            classes = model.classes_

            probability_records = [
                {
                    "Kondisi": str(class_name),
                    "Probabilitas": float(probability)
                }
                for class_name, probability in zip(
                    classes,
                    probabilities
                )
            ]

            result = {
                "prediction": prediction,
                "probabilities": probability_records,
                "sensor": posisi_sensor,
                "timestamp": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
                "input": input_values
            }

            st.session_state.latest_result = result

            st.session_state.history.append({
                "timestamp": result["timestamp"],
                "sensor": posisi_sensor,
                "prediction": prediction
            })

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat melakukan prediksi: {e}"
            )

    with result_col:

        st.markdown(
            '<div class="section-title">✨ Diagnosis Result</div>',
            unsafe_allow_html=True
        )

        if st.session_state.latest_result is not None:

            show_result(
                st.session_state.latest_result
            )

        else:

            st.markdown(
                """
                <div class="result-bubble"
                     style="padding:45px 20px;">
                    <div class="emoji">🔍</div>
                    <h2>Waiting for Analysis</h2>
                    <p>
                        Masukkan data pengukuran dan tekan
                        tombol analisis untuk melihat hasil diagnosis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# 11. HISTORY PAGE
# =========================================================

elif page == "📋 Riwayat Prediksi":

    st.markdown(
        '<div class="section-title">📋 Prediction History</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Daftar hasil prediksi selama sesi aplikasi berjalan."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Hapus Riwayat Prediksi"):

            st.session_state.history = []
            st.session_state.latest_result = None

            st.rerun()

    else:

        st.info(
            "Belum ada riwayat prediksi. "
            "Lakukan diagnosis terlebih dahulu."
        )

# =========================================================
# 12. MODEL INFORMATION PAGE
# =========================================================

elif page == "📈 Informasi Model":

    st.markdown(
        '<div class="section-title">📈 Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="custom-card">
            <h3>🤖 {metadata.get("model_name", "Extra Trees Classifier")}</h3>
            <p><b>Model Type:</b> {metadata.get("model_type", "N/A")}</p>
            <p><b>Target:</b> {metadata.get("target", "N/A")}</p>
            <p><b>Validation Macro F1:</b>
                {metadata.get("validation_macro_f1", 0) * 100:.2f}%
            </p>
            <p><b>Test Macro F1:</b>
                {metadata.get("test_macro_f1", 0) * 100:.2f}%
            </p>
            <p><b>Test Accuracy:</b>
                {metadata.get("test_accuracy", 0) * 100:.2f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧩 Input Features")

    st.dataframe(
        pd.DataFrame({
            "No": range(1, len(FEATURE_COLUMNS) + 1),
            "Feature": FEATURE_COLUMNS
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🏷️ Classification Labels")

    st.write(
        ", ".join(metadata.get("classes", []))
    )

    st.caption(
        "Model menggunakan algoritma Extra Trees "
        "Classifier yang telah dilatih menggunakan "
        "data train dan validation."
    )

# =========================================================
# 13. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;padding:15px;">
        <p style="font-weight:bold;">
            ⚙️ BearingGuard AI
        </p>
        <p style="font-size:12px;">
            Machine Learning Project • Bearing Fault Diagnosis
        </p>
        <p style="font-size:12px;">
            Built with Python & Streamlit | Shinchan Edition 🌟
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
