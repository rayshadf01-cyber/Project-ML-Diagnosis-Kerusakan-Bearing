
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="BearingGuard AI | Shinchan Edition",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"


# =========================================================
# 2. LOAD MODEL DAN METADATA
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


try:
    model = load_model()
    metadata = load_metadata()
    MODEL_READY = True

except Exception as error:
    model = None
    metadata = {}
    MODEL_READY = False
    MODEL_ERROR = str(error)


# =========================================================
# 3. KONFIGURASI FITUR
# =========================================================

DEFAULT_FEATURES = [
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

FEATURE_COLUMNS = metadata.get(
    "features",
    DEFAULT_FEATURES
)

# Sesuaikan dengan encoding posisi_sensor saat training.
SENSOR_MAPPING = metadata.get(
    "sensor_mapping",
    {
        "drive_end": 0,
        "fan_end": 1
    }
)

CLASS_INFO = {
    "normal": {
        "emoji": "🟢",
        "title": "NORMAL",
        "description": "Bearing berada dalam kondisi normal.",
        "recommendation": (
            "Pertahankan jadwal perawatan berkala. "
            "Lakukan pemantauan getaran dan suhu secara rutin."
        )
    },
    "outer_race": {
        "emoji": "🟠",
        "title": "OUTER RACE FAULT",
        "description": "Terindikasi kerusakan pada lintasan luar bearing.",
        "recommendation": (
            "Periksa lintasan luar bearing. "
            "Jadwalkan inspeksi dan pertimbangkan penggantian "
            "bearing jika kerusakan terkonfirmasi."
        )
    },
    "inner_race": {
        "emoji": "🔴",
        "title": "INNER RACE FAULT",
        "description": "Terindikasi kerusakan pada lintasan dalam bearing.",
        "recommendation": (
            "Periksa lintasan dalam, kondisi pelumasan, "
            "dan kemungkinan keausan. Lakukan inspeksi lebih lanjut."
        )
    },
    "bola": {
        "emoji": "🟡",
        "title": "BALL FAULT",
        "description": "Terindikasi kerusakan pada elemen bola bearing.",
        "recommendation": (
            "Periksa elemen bola dan kondisi lintasan bearing. "
            "Lakukan pemeriksaan lanjutan sebelum mesin dioperasikan "
            "kembali secara normal."
        )
    }
}


# =========================================================
# 4. CSS - SHINCHAN THEME
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background-color: #FFF8EC;
    color: #24385F;
}

header[data-testid="stHeader"] {
    background-color: #FFF8EC;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #263B68;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.sidebar-logo {
    text-align: center;
    padding: 15px 5px 20px 5px;
}

.sidebar-logo h2 {
    color: #FFD447;
    font-size: 23px;
    font-weight: 900;
    margin-bottom: 0;
}

.sidebar-logo p {
    color: #FFFFFF;
    font-size: 10px;
    letter-spacing: 1px;
}

/* HEADING */

.main-title {
    color: #263B68;
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 5px;
}

.subtitle {
    color: #71809B;
    font-size: 14px;
    margin-bottom: 25px;
}

/* HERO */

.hero {
    background: linear-gradient(120deg, #263B68, #405B91);
    padding: 32px;
    border-radius: 18px;
    border: 2px solid #FFD447;
    margin-bottom: 25px;
    box-shadow: 0 5px 15px rgba(38, 59, 104, 0.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 32px;
    font-weight: 900;
    margin-bottom: 8px;
}

.hero p {
    color: #F7F2E8 !important;
    font-size: 15px;
    margin-bottom: 0;
}

/* CARDS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    border-color: #E9DCC7;
}

.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E9DCC7;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
    box-shadow: 0 3px 10px rgba(38, 59, 104, 0.05);
}

.metric-label {
    color: #7A879E;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.metric-value {
    color: #263B68;
    font-size: 25px;
    font-weight: 900;
    margin-top: 10px;
}

.metric-caption {
    color: #929BAD;
    font-size: 11px;
    margin-top: 4px;
}

/* RESULT */

.result-card {
    background: #FFFFFF;
    border: 2px solid #FFD447;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.result-label {
    color: #7A879E;
    font-size: 13px;
    font-weight: 800;
}

.result-name {
    color: #263B68;
    font-size: 27px;
    font-weight: 900;
    margin: 12px 0;
}

.result-description {
    color: #65738C;
    font-size: 14px;
}

/* RECOMMENDATION */

.recommendation {
    background: #FFF0BF;
    border-left: 5px solid #FFD447;
    border-radius: 10px;
    padding: 18px;
    color: #263B68;
    margin-top: 15px;
}

/* =====================================================
   BUTTON STYLE - SHINCHAN RED
===================================================== */

.stApp .stButton > button,
.stApp [data-testid="stFormSubmitButton"] button {
    background-color: #E72D46 !important;
    color: #FFFFFF !important;
    border: 2px solid #C91F38 !important;
    border-radius: 10px !important;
    min-height: 46px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

/* Tulisan di dalam tombol */
.stApp .stButton > button *,
.stApp [data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

/* Hover tombol */
.stApp .stButton > button:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover {
    background-color: #C91F38 !important;
    border-color: #A91930 !important;
}

/* =====================================================
   INPUT LABEL - DARK NAVY
===================================================== */

/* Label seluruh widget */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] *,
.stApp .stNumberInput label,
.stApp .stSelectbox label {
    color: #263B68 !important;
    font-size: 14px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

/* Label di dalam form */
.stApp [data-testid="stForm"] label,
.stApp [data-testid="stForm"] label * {
    color: #263B68 !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

/* INPUT FIELD */

.stApp .stNumberInput input,
.stApp .stTextInput input {
    color: #263B68 !important;
    background-color: #FFFFFF !important;
    font-weight: 700 !important;
    border: 1px solid #C8D0DF !important;
}

/* DROPDOWN SENSOR */

.stApp .stSelectbox div[data-baseweb="select"] {
    border-radius: 8px !important;
    background-color: #FFFFFF !important;
    color: #263B68 !important;
    font-weight: 700 !important;
}

/* Fokus input */

.stApp .stNumberInput input:focus,
.stApp .stTextInput input:focus {
    border: 2px solid #E72D46 !important;
    box-shadow: 0 0 0 2px rgba(231, 45, 70, 0.12) !important;
}

/* GENERAL HEADING */

h1, h2, h3 {
    color: #263B68;
    font-weight: 900;
}

hr {
    border-color: #E9DCC7;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #8A94A8;
    font-size: 12px;
    padding: 25px 5px;
}

.small-note {
    color: #7A879E;
    font-size: 12px;
}

/* RESPONSIVE */

@media (max-width: 768px) {
    .main-title {
        font-size: 25px;
    }

    .hero {
        padding: 22px;
    }

    .hero h1 {
        font-size: 25px;
    }

    .metric-value {
        font-size: 21px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# =========================================================
# 6. FUNGSI BANTUAN
# =========================================================

def format_class_name(value):
    value = str(value).lower()

    if value in CLASS_INFO:
        return CLASS_INFO[value]["title"]

    return value.replace("_", " ").upper()


def get_class_info(value):
    return CLASS_INFO.get(
        str(value).lower(),
        {
            "emoji": "⚙️",
            "title": format_class_name(value),
            "description": "Kondisi bearing teridentifikasi.",
            "recommendation": (
                "Lakukan pemeriksaan kondisi bearing "
                "untuk memastikan hasil diagnosis."
            )
        }
    )


def show_hero():
    st.markdown("""
    <div class="hero">
        <h1>⚙️ BearingGuard AI</h1>
        <p>
            Intelligent Bearing Fault Diagnosis System
        </p>
        <p>
            Platform klasifikasi kondisi bearing berdasarkan
            parameter pengukuran getaran mesin.
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_metric(label, value, caption=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def get_metric(key, default):
    value = metadata.get(key, default)

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def make_input_data(values):
    """
    Menyusun data input sesuai urutan fitur model.
    """

    input_data = {}

    for feature in FEATURE_COLUMNS:
        if feature not in values:
            raise ValueError(
                f"Fitur {feature} belum tersedia di form input."
            )

        input_data[feature] = values[feature]

    return pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )


def show_prediction_result(result):
    info = get_class_info(result["prediction"])

    st.markdown("### ✨ Diagnosis Result")

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">HASIL KLASIFIKASI BEARING</div>
        <div style="font-size:48px; margin-top:12px;">
            {info["emoji"]}
        </div>
        <div class="result-name">
            {info["title"]}
        </div>
        <div class="result-description">
            {info["description"]}
        </div>
        <div style="
            background:#FFF0BF;
            border-radius:10px;
            padding:12px;
            margin-top:18px;
        ">
            <div style="color:#65738C;font-size:12px;">
                Tingkat Keyakinan Model
            </div>
            <div style="
                color:#263B68;
                font-size:28px;
                font-weight:900;
            ">
                {result["confidence"]:.2f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Maintenance Recommendation")

    st.info(info["recommendation"])

    st.warning(
        "Hasil klasifikasi merupakan prediksi model AI. "
        "Lakukan pemeriksaan teknis untuk mengonfirmasi hasil diagnosis."
    )

    st.markdown("### 📊 Prediction Confidence")

    probability_df = pd.DataFrame(
        result["probabilities"]
    )

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=True
    )

    st.bar_chart(
        probability_df.set_index("Condition")["Probability"],
        horizontal=True,
        height=350
    )

    st.markdown("### 📋 Probability Details")

    display_df = probability_df.copy()

    display_df["Probability"] = (
        display_df["Probability"].map(
            lambda x: f"{x:.2f}%"
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:38px;">⚙️</div>
        <h2>BearingGuard</h2>
        <p>SMART BEARING DIAGNOSTIC SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📌 MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Diagnosis",
            "Riwayat Prediksi",
            "Informasi Model"
        ],
        label_visibility="collapsed",
        key="navigation"
    )

    st.session_state.page = page

    st.divider()

    st.markdown("""
    <div style="
        background:#344D7D;
        padding:16px;
        border-radius:12px;
        text-align:center;
    ">
        <div style="font-size:30px;">🧒🏻 🐶</div>
        <div style="
            color:#FFD447;
            font-size:14px;
            font-weight:900;
        ">
            SHINCHAN MODE
        </div>
        <div style="font-size:11px;color:white;">
            Keep your machine happy and healthy!
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.caption("BearingGuard AI | Machine Learning Project")


# =========================================================
# 8. CEK MODEL
# =========================================================

if not MODEL_READY:

    st.error("Model belum berhasil dimuat.")

    st.write("Detail kesalahan:")

    st.code(MODEL_ERROR)

    st.info(
        "Pastikan file model.joblib berada di folder models "
        "dan file metadata.json tersedia."
    )

    st.stop()


# =========================================================
# 9. DASHBOARD
# =========================================================

if page == "Dashboard":

    show_hero()

    st.markdown("## 📊 System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric(
            "Model Algorithm",
            metadata.get(
                "model_name",
                "Extra Trees Classifier"
            ),
            "Machine Learning Algorithm"
        )

    with col2:
        show_metric(
            "Validation Macro F1",
            f"{get_metric('validation_macro_f1', 0.8047) * 100:.2f}%",
            "Validation Performance"
        )

    with col3:
        show_metric(
            "Test Accuracy",
            f"{get_metric('test_accuracy', 0.8418) * 100:.2f}%",
            "Testing Performance"
        )

    st.divider()

    st.markdown("## 🧩 Bearing Condition Classes")

    class_columns = st.columns(4)

    class_cards = [
        ("🟢", "NORMAL", "Kondisi bearing normal."),
        ("🟠", "OUTER RACE", "Kerusakan lintasan luar."),
        ("🔴", "INNER RACE", "Kerusakan lintasan dalam."),
        ("🟡", "BALL FAULT", "Kerusakan elemen bola.")
    ]

    for column, card in zip(class_columns, class_cards):

        emoji, title, description = card

        with column:
            st.markdown(f"""
            <div class="metric-card" style="
                min-height:150px;
                text-align:center;
            ">
                <div style="font-size:30px;">{emoji}</div>
                <div style="
                    color:#263B68;
                    font-weight:900;
                    margin-top:8px;
                ">
                    {title}
                </div>
                <div style="
                    color:#7A879E;
                    font-size:12px;
                    margin-top:8px;
                ">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## 📈 Recent Predictions")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df.tail(5).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Belum ada riwayat prediksi. "
            "Silakan lakukan diagnosis pertama."
        )

    st.divider()

    st.markdown(
        "<div class='footer'>"
        "⚙️ BearingGuard AI<br>"
        "Machine Learning Project · Bearing Fault Diagnosis<br>"
        "Shinchan-inspired Edition"
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 10. HALAMAN DIAGNOSIS
# =========================================================

elif page == "Diagnosis":

    show_hero()

    st.markdown("## 🔍 Bearing Diagnosis")

    st.write(
        "Masukkan parameter pengukuran untuk menganalisis "
        "kondisi bearing menggunakan model Machine Learning."
    )

    if "posisi_sensor" not in FEATURE_COLUMNS:
        st.error(
            "Fitur posisi_sensor tidak ditemukan. "
            "Periksa metadata dan fitur model."
        )
        st.stop()

    # FORM INPUT

    with st.form("bearing_diagnosis_form"):

        st.markdown("### 📍 Sensor Position")

        sensor_label = st.selectbox(
            "Pilih posisi sensor",
            list(SENSOR_MAPPING.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )

        st.markdown("### ⚙️ Operating Parameters")

        left, right = st.columns(2)

        values = {}

        with left:

            values["kecepatan_rpm"] = st.number_input(
                "Kecepatan (RPM)",
                min_value=0.0,
                value=2300.0,
                step=50.0
            )

            values["beban_kN"] = st.number_input(
                "Beban (kN)",
                min_value=0.0,
                value=2.30,
                step=0.1
            )

            values["suhu_housing_C"] = st.number_input(
                "Suhu Housing (°C)",
                min_value=0.0,
                value=50.0,
                step=1.0
            )

            values["rms_g"] = st.number_input(
                "RMS (g)",
                min_value=0.0,
                value=3.40,
                step=0.1,
                format="%.4f"
            )

            values["puncak_g"] = st.number_input(
                "Puncak (g)",
                min_value=0.0,
                value=14.0,
                step=0.1,
                format="%.4f"
            )

            values["kurtosis"] = st.number_input(
                "Kurtosis",
                min_value=0.0,
                value=3.2,
                step=0.1,
                format="%.4f"
            )

        with right:

            values["crest_factor"] = st.number_input(
                "Crest Factor",
                min_value=0.0,
                value=2.5,
                step=0.1,
                format="%.4f"
            )

            values["energi_bpfo"] = st.number_input(
                "Energy BPFO",
                min_value=0.0,
                value=0.004,
                step=0.001,
                format="%.6f"
            )

            values["energi_bpfi"] = st.number_input(
                "Energy BPFI",
                min_value=0.0,
                value=0.003,
                step=0.001,
                format="%.6f"
            )

            values["energi_bsf"] = st.number_input(
                "Energy BSF",
                min_value=0.0,
                value=0.002,
                step=0.001,
                format="%.6f"
            )

            values["frekuensi_dominan_hz"] = st.number_input(
                "Frekuensi Dominan (Hz)",
                min_value=0.0,
                value=150.0,
                step=1.0,
                format="%.4f"
            )

        submitted = st.form_submit_button(
            "🔍 ANALYZE BEARING CONDITION",
            use_container_width=True
        )

    # PROSES PREDIKSI

    if submitted:

        try:

            values["posisi_sensor"] = SENSOR_MAPPING[
                sensor_label
            ]

            input_df = make_input_data(values)

            prediction = model.predict(input_df)[0]

            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "Model tidak mendukung predict_proba."
                )

            probabilities = model.predict_proba(
                input_df
            )[0]

            classes = list(model.classes_)

            probability_records = []

            for class_name, probability in zip(
                classes,
                probabilities
            ):

                probability_records.append({
                    "Condition": format_class_name(class_name),
                    "Probability": float(probability) * 100
                })

            confidence = max(probabilities) * 100

            result = {
                "prediction": str(prediction),
                "confidence": float(confidence),
                "probabilities": probability_records,
                "timestamp": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
                "sensor": sensor_label
            }

            st.session_state.latest_result = result

            st.session_state.history.append({
                "Waktu": result["timestamp"],
                "Sensor": sensor_label,
                "Prediction": format_class_name(prediction),
                "Confidence": f"{confidence:.2f}%"
            })

            st.success("Diagnosis berhasil dilakukan!")

        except Exception as error:

            st.error("Terjadi kesalahan saat melakukan prediksi.")

            st.exception(error)

    # TAMPILKAN HASIL PREDIKSI

    if st.session_state.latest_result is not None:

        st.divider()

        show_prediction_result(
            st.session_state.latest_result
        )


# =========================================================
# 11. HALAMAN RIWAYAT
# =========================================================

elif page == "Riwayat Prediksi":

    show_hero()

    st.markdown("## 📜 Prediction History")

    st.write(
        "Daftar hasil prediksi yang telah dilakukan "
        "selama sesi aplikasi ini."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 📊 Prediction Summary")

        summary = (
            history_df["Prediction"]
            .value_counts()
            .reset_index()
        )

        summary.columns = [
            "Condition",
            "Total Prediction"
        ]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.bar_chart(
                summary.set_index("Condition"),
                horizontal=True,
                height=300
            )

        with col2:
            show_metric(
                "Total Prediction",
                len(history_df),
                "All predictions in this session"
            )

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            st.session_state.history = []
            st.session_state.latest_result = None

            st.rerun()

    else:

        st.info(
            "Belum ada riwayat prediksi. "
            "Silakan lakukan diagnosis terlebih dahulu."
        )


# =========================================================
# 12. HALAMAN INFORMASI MODEL
# =========================================================

elif page == "Informasi Model":

    show_hero()

    st.markdown("## 🧠 Model Information")

    st.markdown("### Model Overview")

    col1, col2 = st.columns(2)

    with col1:

        show_metric(
            "Algorithm",
            metadata.get(
                "model_name",
                "Extra Trees Classifier"
            ),
            "Classification Algorithm"
        )

    with col2:

        show_metric(
            "Target Variable",
            metadata.get("target", "kondisi"),
            "Prediction Target"
        )

    st.divider()

    st.markdown("### 📈 Model Performance")

    performance = pd.DataFrame({
        "Metric": [
            "Validation Macro F1",
            "Test Macro F1",
            "Test Accuracy"
        ],
        "Score": [
            get_metric("validation_macro_f1", 0.8047),
            get_metric("test_macro_f1", 0.7991),
            get_metric("test_accuracy", 0.8418)
        ]
    })

    performance["Percentage"] = (
        performance["Score"] * 100
    ).map(lambda x: f"{x:.2f}%")

    st.dataframe(
        performance[["Metric", "Percentage"]],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### 🧩 Input Features")

    feature_df = pd.DataFrame({
        "No": range(1, len(FEATURE_COLUMNS) + 1),
        "Feature": FEATURE_COLUMNS
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### 🏷️ Classification Labels")

    model_classes = getattr(
        model,
        "classes_",
        list(CLASS_INFO.keys())
    )

    class_df = pd.DataFrame({
        "Class": [
            format_class_name(x)
            for x in model_classes
        ]
    })

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### 🔧 Technical Information")

    st.write("**Model File:** `models/model.joblib`")

    st.write("**Metadata File:** `models/metadata.json`")

    st.write(
        "**Model Algorithm:**",
        metadata.get(
            "model_name",
            "Extra Trees Classifier"
        )
    )

    st.write(
        "**Target:**",
        metadata.get("target", "kondisi")
    )

    st.write(
        "**Number of Features:**",
        len(FEATURE_COLUMNS)
    )

    st.write("**Sensor Mapping:**")

    st.json(SENSOR_MAPPING)


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <div style="font-size:25px;">⚙️ 🧒🏻 🐶</div>
    <div style="
        color:#263B68;
        font-size:16px;
        font-weight:900;
    ">
        BearingGuard AI
    </div>
    Machine Learning Project · Bearing Fault Diagnosis
    <br>
    Shinchan-inspired Edition
</div>
""", unsafe_allow_html=True)
