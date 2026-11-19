
import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="BearingGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"

SENSOR_MAPPING = {
    "drive_end": 0,
    "fan_end": 1
}

FEATURE_COLUMNS = [
    "posisi_sensor",
    "kecepatan_rpm",
    "beban_kN",
    "suhu_housing_C",
    "rms_g",
    "puncak_g",
    "kurtosis",
    "crest_factor",
    "energi_bpfo",
    "energi_bpfi",
    "energi_bsf",
    "frekuensi_dominan_hz"
]

# =====================================================
# 2. CUSTOM CSS - SHINCHAN THEME
# =====================================================

st.markdown("""
<style>

/* GENERAL */
.stApp {
    background-color: #FFF8E9;
    color: #202E52;
}

[data-testid="stAppViewContainer"] {
    background-color: #FFF8E9;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #202E52;
    border-right: 4px solid #F5C542;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFD65A;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #202E52 !important;
    font-weight: 900 !important;
}

p, li {
    color: #30394F;
}

/* LABEL */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
    color: #202E52 !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

/* INPUT */
div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 2px solid #CBD2DF !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    color: #17233D !important;
    -webkit-text-fill-color: #17233D !important;
    background: #FFFFFF !important;
    font-weight: 700 !important;
}

/* SELECT */
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 2px solid #CBD2DF !important;
}

div[data-baseweb="select"] * {
    color: #17233D !important;
    -webkit-text-fill-color: #17233D !important;
    font-weight: 700 !important;
}

/* BUTTON */
.stButton > button,
.stFormSubmitButton > button {
    background: #D62839 !important;
    color: #FFFFFF !important;
    border: 2px solid #B91F30 !important;
    border-radius: 12px !important;
    font-weight: 900 !important;
    min-height: 45px;
    transition: 0.2s;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #A91F2E !important;
    border-color: #A91F2E !important;
    transform: translateY(-2px);
}

/* SIDEBAR RADIO */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #35476F;
    border-radius: 10px;
    padding: 9px;
    margin-bottom: 7px;
}

/* CARDS */
.custom-card {
    background: #FFFFFF;
    border: 2px solid #D8DFEA;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 14px rgba(32,46,82,0.08);
    margin-bottom: 12px;
}

.metric-card {
    background: #FFFFFF;
    border: 2px solid #D8DFEA;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.metric-label {
    color: #526078;
    font-size: 13px;
    font-weight: 800;
}

.metric-value {
    color: #202E52;
    font-size: 27px;
    font-weight: 900;
}

/* RESULT CARD */
.result-card {
    background: #FFFFFF;
    border: 3px solid #F5C542;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(32,46,82,0.10);
}

.result-title {
    color: #526078;
    font-size: 14px;
    font-weight: 800;
}

.result-value {
    font-size: 32px;
    font-weight: 1000;
    margin-top: 8px;
}

/* ALERT */
[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stAlert"] p {
    color: #202944 !important;
    font-weight: 600;
}

/* EXPANDER */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 2px solid #CBD2DF;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #202E52 !important;
    font-weight: 800 !important;
}

/* DIVIDER */
hr {
    border-color: #D8DFEA !important;
}

/* MOBILE */
@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem;
    }

    h1 {
        font-size: 27px !important;
    }

    h2 {
        font-size: 22px !important;
    }

    .metric-value {
        font-size: 21px;
    }

    .result-value {
        font-size: 26px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL
# =====================================================

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

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# =====================================================
# 4. SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# =====================================================
# 5. CLASS INFORMATION
# =====================================================

CLASS_INFO = {

    "normal": {
        "title": "NORMAL",
        "emoji": "💚",
        "color": "#168A52",
        "description": "Bearing berada dalam kondisi normal.",
        "recommendation": (
            "Lanjutkan pemantauan kondisi mesin dan "
            "lakukan perawatan berkala."
        )
    },

    "outer_race": {
        "title": "OUTER RACE",
        "emoji": "🔴",
        "color": "#D62839",
        "description": "Terindikasi kerusakan pada outer race.",
        "recommendation": (
            "Lakukan inspeksi pada bagian outer race. "
            "Pertimbangkan penggantian bearing apabila "
            "kerusakan telah terkonfirmasi."
        )
    },

    "inner_race": {
        "title": "INNER RACE",
        "emoji": "🟠",
        "color": "#E67E22",
        "description": "Terindikasi kerusakan pada inner race.",
        "recommendation": (
            "Jadwalkan pemeriksaan lebih lanjut pada inner race "
            "dan siapkan komponen pengganti."
        )
    },

    "bola": {
        "title": "BALL ELEMENT",
        "emoji": "🟡",
        "color": "#B8860B",
        "description": "Terindikasi kerusakan pada elemen bola.",
        "recommendation": (
            "Periksa kondisi elemen bola dan lakukan inspeksi "
            "bearing sebelum menentukan tindakan perbaikan."
        )
    }
}


# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 5px;">
            <div style="font-size:48px;">⚙️</div>
            <h2 style="color:#FFD65A !important; margin:0;">
                BearingGuard
            </h2>
            <p style="font-size:12px; margin-top:8px;">
                SMART BEARING DIAGNOSTIC SYSTEM
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧭 MAIN MENU")

    page = st.radio(
        "Navigasi",
        [
            "🏠 Dashboard",
            "🔧 Diagnosis",
            "📋 Riwayat Prediksi",
            "📊 Informasi Model"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        """
        <div style="
            background:#35476F;
            padding:17px;
            border-radius:14px;
            border:1px solid #50638E;
        ">
            <h3 style="color:#FFD65A !important;">
                🌟 Shinchan Mode
            </h3>
            <p style="font-size:13px;">
                Keep your machines happy and healthy!
            </p>
            <p style="font-size:26px; text-align:center;">
                🧒🏻 🔧 ⚙️
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption("Machine Learning Project | UPI")
    st.caption("Bearing Fault Diagnosis")


# =====================================================
# 7. HEADER
# =====================================================

st.markdown(
    """
    <div style="
        background:#202E52;
        border-radius:18px;
        padding:25px;
        border-bottom:6px solid #F5C542;
        margin-bottom:25px;
    ">
        <div style="
            color:#FFD65A;
            font-size:12px;
            font-weight:900;
            letter-spacing:2px;
        ">
            INDUSTRIAL AI • PREDICTIVE MAINTENANCE
        </div>

        <h1 style="
            color:#FFFFFF !important;
            margin:8px 0;
        ">
            ⚙️ BearingGuard AI
        </h1>

        <p style="
            color:#FFFFFF;
            margin:0;
            font-size:15px;
        ">
            Intelligent Bearing Fault Diagnosis System
        </p>

        <p style="
            color:#E5EAF5;
            font-size:13px;
            margin-top:8px;
        ">
            Platform klasifikasi kondisi bearing berdasarkan
            parameter pengukuran getaran mesin.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 8. DASHBOARD PAGE
# =====================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard Overview")

    st.write(
        "Selamat datang di BearingGuard AI. "
        "Sistem ini membantu mengidentifikasi kondisi bearing "
        "berdasarkan data pengukuran getaran."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">MODEL ALGORITHM</div>
                <div class="metric-value">Extra Trees</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TEST MACRO F1</div>
                <div class="metric-value">
                    {metadata['test_macro_f1'] * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TEST ACCURACY</div>
                <div class="metric-value">
                    {metadata['test_accuracy'] * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.subheader("🔍 Available Diagnosis")

    cards = st.columns(4)

    for i, (key, info) in enumerate(CLASS_INFO.items()):

        with cards[i]:
            st.markdown(
                f"""
                <div class="custom-card"
                    style="text-align:center; min-height:150px;">
                    <div style="font-size:35px;">
                        {info['emoji']}
                    </div>
                    <h4 style="color:{info['color']} !important;">
                        {info['title']}
                    </h4>
                    <p style="font-size:12px;">
                        {info['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info(
        "Pilih menu Diagnosis pada sidebar untuk mulai "
        "melakukan prediksi kondisi bearing."
    )


# =====================================================
# 9. DIAGNOSIS PAGE
# =====================================================

elif page == "🔧 Diagnosis":

    st.header("🔧 Input Measurement")
    st.caption(
        "Masukkan parameter pengukuran untuk menganalisis "
        "kondisi bearing."
    )

    # Two main panels
    input_col, result_col = st.columns(
        [1.15, 0.85],
        gap="large"
    )

    with input_col:

        st.markdown(
            """
            <div class="custom-card">
                <h3>📥 Measurement Data</h3>
                <p>Masukkan data sensor dan karakteristik getaran.</p>
            </div>
            """,
            unsafe_allow_html=True
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
                    value=2.3,
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

            with c2:

                puncak_g = st.number_input(
                    "Puncak (g)",
                    min_value=0.0,
                    value=1.4368,
                    step=0.01,
                    format="%.4f"
                )

                frekuensi_dominan_hz = st.number_input(
                    "Frekuensi Dominan (Hz)",
                    min_value=0.0,
                    value=38.7,
                    step=0.1
                )

            st.markdown("### 📈 Vibration Characteristics")

            c3, c4 = st.columns(2)

            with c3:

                kurtosis = st.number_input(
                    "Kurtosis",
                    value=2.209,
                    step=0.01,
                    format="%.4f"
                )

                crest_factor = st.number_input(
                    "Crest Factor",
                    min_value=0.0,
                    value=2.905,
                    step=0.01,
                    format="%.4f"
                )

                energi_bpfo = st.number_input(
                    "Energi BPFO",
                    min_value=0.0,
                    value=0.0042,
                    step=0.001,
                    format="%.5f"
                )

            with c4:

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

            st.divider()

            submitted = st.form_submit_button(
                "🔍 ANALYZE BEARING CONDITION",
                use_container_width=True
            )

    # =================================================
    # PREDICTION PROCESS
    # =================================================

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

        input_data = pd.DataFrame([input_values])

        # Match training feature order
        expected_features = metadata.get(
            "features",
            FEATURE_COLUMNS
        )

        input_data = input_data[expected_features]

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        probability_df = pd.DataFrame({
            "Kondisi": model.classes_,
            "Probabilitas": probabilities
        })

        probability_df["Persentase"] = (
            probability_df["Probabilitas"] * 100
        )

        probability_df = probability_df.sort_values(
            "Probabilitas",
            ascending=False
        )

        result = {
            "prediction": prediction,
            "probabilities": probability_df.to_dict("records"),
            "sensor": posisi_sensor,
            "timestamp": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "input": input_values
        }

        st.session_state.latest_result = result

        st.session_state.history.insert(0, result)

    # =================================================
    # RESULT PANEL
    # =================================================

    with result_col:

        st.markdown(
            """
            <div class="custom-card">
                <h3>✨ Diagnosis Result</h3>
                <p>Hasil analisis kondisi bearing.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        result = st.session_state.latest_result

        if result is None:

            st.markdown(
                """
                <div class="result-card"
                    style="border-color:#CBD2DF; padding:45px 20px;">
                    <div style="font-size:55px;">🔎</div>
                    <h2 style="color:#202E52 !important;">
                        Waiting for Analysis
                    </h2>
                    <p>
                        Masukkan data pengukuran dan tekan tombol
                        analisis untuk melihat hasil diagnosis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            prediction = result["prediction"]

            info = CLASS_INFO.get(
                prediction,
                CLASS_INFO["normal"]
            )

            st.markdown(
                f"""
                <div class="result-card"
                    style="border-color:{info['color']};">
                    <div style="font-size:48px;">
                        {info['emoji']}
                    </div>
                    <div class="result-title">
                        PREDICTED BEARING CONDITION
                    </div>
                    <div class="result-value"
                        style="color:{info['color']};">
                        {info['title']}
                    </div>
                    <p>{info['description']}</p>
                    <p style="font-size:12px;">
                        Sensor: {result['sensor']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown("### 📊 Prediction Confidence")

            chart_data = pd.DataFrame(
                result["probabilities"]
            ).set_index("Kondisi")

            st.bar_chart(
                chart_data["Persentase"],
                horizontal=True
            )

            st.markdown("### 📋 Probability Details")

            display_df = pd.DataFrame(
                result["probabilities"]
            )

            display_df["Persentase"] = (
                display_df["Persentase"].map(
                    lambda x: f"{x:.2f}%"
                )
            )

            st.dataframe(
                display_df[["Kondisi", "Persentase"]],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 🛠️ Maintenance Recommendation")

            st.info(info["recommendation"])

            st.warning(
                "Lakukan inspeksi teknis lanjutan sebelum "
                "menentukan tindakan perbaikan."
            )

            st.caption(
                f"Waktu prediksi: {result['timestamp']}"
            )

    st.divider()

    st.caption(
        "BearingGuard AI merupakan alat bantu analisis "
        "berbasis machine learning, bukan pengganti inspeksi teknis."
    )


# =====================================================
# 10. HISTORY PAGE
# =====================================================

elif page == "📋 Riwayat Prediksi":

    st.header("📋 Prediction History")

    st.write(
        "Daftar hasil diagnosis yang dilakukan selama "
        "sesi aplikasi ini berlangsung."
    )

    if len(st.session_state.history) == 0:

        st.info("Belum ada riwayat prediksi.")

    else:

        history_data = []

        for item in st.session_state.history:

            history_data.append({
                "Waktu": item["timestamp"],
                "Posisi Sensor": item["sensor"],
                "Hasil Prediksi": item["prediction"].upper()
            })

        history_df = pd.DataFrame(history_data)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"Total prediksi: {len(history_df)}"
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Riwayat CSV",
            data=csv,
            file_name="bearing_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "🗑️ Hapus Riwayat",
            use_container_width=True
        ):

            st.session_state.history = []
            st.session_state.latest_result = None
            st.rerun()


# =====================================================
# 11. MODEL INFORMATION PAGE
# =====================================================

elif page == "📊 Informasi Model":

    st.header("📊 Model Information")

    st.write(
        "Informasi model machine learning yang digunakan "
        "untuk melakukan klasifikasi kondisi bearing."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="custom-card">
                <h3>🤖 Model Details</h3>
                <p><b>Model:</b> {metadata['model_name']}</p>
                <p><b>Algorithm:</b> {metadata['model_type']}</p>
                <p><b>Target:</b> {metadata['target']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="custom-card">
                <h3>📈 Evaluation Results</h3>
                <p><b>Validation Macro F1:</b>
                {metadata['validation_macro_f1']:.4f}</p>
                <p><b>Test Macro F1:</b>
                {metadata['test_macro_f1']:.4f}</p>
                <p><b>Test Accuracy:</b>
                {metadata['test_accuracy']:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("🎯 Classification Classes")

    for key, info in CLASS_INFO.items():

        st.markdown(
            f"""
            <div class="custom-card"
                style="border-left:7px solid {info['color']};">
                <h4 style="color:{info['color']} !important;">
                    {info['emoji']} {info['title']}
                </h4>
                <p>{info['description']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("🧩 Model Features")

    feature_df = pd.DataFrame({
        "No": range(1, len(metadata["features"]) + 1),
        "Feature": metadata["features"]
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("⚙️ Best Hyperparameters")

    if "best_params" in metadata:

        params_df = pd.DataFrame(
            list(metadata["best_params"].items()),
            columns=["Parameter", "Value"]
        )

        st.dataframe(
            params_df,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Model dilatih menggunakan algoritma Extra Trees "
        "Classifier dengan konfigurasi hyperparameter terbaik."
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.markdown(
    """
    <hr>

    <div style="text-align:center; padding:15px;">
        <h4 style="color:#202E52 !important;">
            ⚙️ BearingGuard AI
        </h4>

        <p style="font-size:12px;">
            Machine Learning-Based Bearing Fault Diagnosis
        </p>

        <p style="font-size:11px;">
            Developed for Machine Learning Project • UPI
        </p>

        <p style="font-size:22px;">
            🧒🏻 ❤️ ⚙️
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
