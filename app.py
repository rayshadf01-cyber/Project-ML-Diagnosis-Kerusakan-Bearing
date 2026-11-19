
import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="BearingGuard AI | Shinchan Edition",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

:root {
    --navy: #26365D;
    --navy2: #354A78;
    --cream: #FFF8EB;
    --yellow: #FFD43B;
    --red: #D9233F;
    --ink: #26365D;
    --muted: #64748B;
    --line: #E8D8BD;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, label {
    color: var(--ink);
}

h1, h2, h3 {
    font-weight: 850 !important;
}

[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 4px solid var(--yellow);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: var(--navy2);
    border: 1px solid #ffffff20;
    border-radius: 12px;
    padding: 9px 12px;
    margin: 5px 0;
}

.hero {
    background: linear-gradient(120deg, #26365D, #405A91);
    border: 3px solid var(--yellow);
    border-radius: 22px;
    padding: 25px 28px;
    margin-bottom: 22px;
    box-shadow: 0 8px 22px #26365d20;
}

.hero .eyebrow {
    color: var(--yellow);
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
}

.hero h1 {
    color: white !important;
    font-size: clamp(26px, 3vw, 38px);
    margin: 7px 0;
}

.hero p {
    color: white !important;
}

.section-heading {
    color: var(--navy);
    font-size: clamp(20px, 2vw, 26px);
    font-weight: 900;
    margin: 8px 0 14px;
}

.panel, .metric-card, .result-card, .class-card {
    background: white;
    border: 2px solid var(--line);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 5px 16px #26365d10;
}

.metric-card {
    text-align: center;
    min-height: 112px;
    border-color: #DCE4F2;
}

.metric-label {
    color: #526487;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .5px;
}

.metric-value {
    color: var(--navy);
    font-size: clamp(20px, 2.4vw, 28px);
    font-weight: 950;
    margin-top: 5px;
}

.class-card {
    text-align: center;
    min-height: 145px;
}

.class-emoji {
    font-size: 34px;
}

.class-name {
    font-weight: 900;
    color: var(--navy);
    margin: 5px 0;
}

.class-desc {
    color: #526487;
    font-size: 13px;
}

.result-card {
    border: 3px solid var(--yellow);
    text-align: center;
    padding: 30px 18px;
    min-height: 270px;
}

.result-emoji {
    font-size: 64px;
    line-height: 1.2;
}

.result-kicker {
    color: #526487;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .7px;
    margin-top: 10px;
}

.result-name {
    font-size: clamp(28px, 3vw, 40px);
    font-weight: 950;
    margin: 12px 0;
}

.recommendation {
    background: #EAF1FF;
    border-left: 6px solid #405A91;
    border-radius: 13px;
    padding: 18px;
    color: #26365D;
    line-height: 1.7;
    font-size: 16px;
}

.warning-box {
    background: #FFF1C2;
    border-left: 6px solid #E7AD18;
    border-radius: 13px;
    padding: 16px;
    color: #59430B;
    line-height: 1.6;
    margin-top: 12px;
}

.mascot-card {
    background: #354A78;
    border: 1px solid #6379A5;
    border-radius: 16px;
    padding: 14px;
    margin-top: 12px;
}

.mascot {
    font-size: 40px;
    text-align: center;
    margin: 2px 0 6px;
}

.mascot-caption {
    color: white;
    text-align: center;
    font-size: 12px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: var(--red) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 12px !important;
    font-weight: 900 !important;
    min-height: 48px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #B81732 !important;
    transform: translateY(-1px);
}

.stNumberInput label,
.stSelectbox label {
    color: var(--navy) !important;
    font-weight: 800 !important;
}

.stNumberInput input {
    background: white !important;
    color: #26365D !important;
    border: 1px solid #AAB9D1 !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: white !important;
    border-radius: 10px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #DCE4F2;
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
}

hr {
    border-color: #D8C9B2 !important;
}

/* RESPONSIVE */

@media(max-width: 900px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }

    .hero {
        padding: 18px;
    }

    .panel, .metric-card, .result-card {
        padding: 14px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"File {MODEL_PATH} tidak ditemukan."
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"File {METADATA_PATH} tidak ditemukan."
        )

    model = joblib.load(MODEL_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()

except Exception as exc:

    st.error(
        f"Gagal memuat model atau metadata: {exc}"
    )

    st.info(
        "Pastikan model.joblib dan metadata.json "
        "berada di folder models/."
    )

    st.stop()


FEATURE_COLUMNS = metadata.get("features", [])

if not FEATURE_COLUMNS:
    st.error(
        "Daftar fitur tidak ditemukan pada metadata.json."
    )
    st.stop()


# ==========================================
# CONFIGURATION
# ==========================================

# Pastikan mapping sama dengan proses training.
SENSOR_MAPPING = {
    "drive_end": 0,
    "fan_end": 1
}

CLASS_INFO = {

    "normal": {
        "emoji": "💚",
        "color": "#16834A",
        "title": "NORMAL",
        "recommendation":
            "Bearing terdeteksi dalam kondisi normal. "
            "Lanjutkan pemantauan getaran dan "
            "perawatan berkala."
    },

    "outer_race": {
        "emoji": "🔴",
        "color": "#D9233F",
        "title": "OUTER RACE",
        "recommendation":
            "Terindikasi kerusakan pada outer race. "
            "Lakukan inspeksi dan pertimbangkan "
            "penggantian bearing."
    },

    "inner_race": {
        "emoji": "🟠",
        "color": "#D97706",
        "title": "INNER RACE",
        "recommendation":
            "Terindikasi kerusakan pada inner race. "
            "Jadwalkan pemeriksaan dan siapkan "
            "komponen pengganti."
    },

    "bola": {
        "emoji": "🟣",
        "color": "#7851A9",
        "title": "BALL FAULT",
        "recommendation":
            "Terindikasi kerusakan elemen bola. "
            "Periksa kondisi bearing dan siapkan "
            "komponen pengganti."
    }
}


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <div style='text-align:center;padding:14px 4px 18px'>
            <div style='font-size:48px'>⚙️</div>
            <h2 style='color:white!important;margin:4px 0'>
                BearingGuard
            </h2>
            <p style='color:#FFD43B!important;
            font-size:12px;font-weight:800'>
                SMART BEARING DIAGNOSTIC SYSTEM
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🏠 MAIN MENU")

    page = st.radio(
        "Navigasi",
        [
            "📊 Dashboard",
            "🔧 Diagnosis",
            "📋 Riwayat Prediksi",
            "📈 Informasi Model"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class='mascot-card'>
            <div class='mascot'>🧒🏻 🐶</div>
            <div style='color:#FFD43B;text-align:center;
            font-weight:900'>
                SHINCHAN MODE
            </div>
            <div class='mascot-caption'>
                Keep your machines happy and healthy!
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption(
        "BearingGuard AI | Machine Learning Project"
    )


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <div class='hero'>
        <div class='eyebrow'>
            INDUSTRIAL AI • PREDICTIVE MAINTENANCE
        </div>

        <h1>⚙️ BearingGuard AI</h1>

        <p>
            <b>Intelligent Bearing Fault Diagnosis System</b>
        </p>

        <p>
            Platform klasifikasi kondisi bearing
            berdasarkan parameter pengukuran getaran mesin.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# DISPLAY RESULT FUNCTION
# ==========================================

def show_result(result):

    prediction = result["prediction"]

    info = CLASS_INFO.get(
        prediction,
        {
            "emoji": "⚙️",
            "color": "#26365D",
            "title": str(prediction).upper(),
            "recommendation":
                "Lakukan pemeriksaan teknis lebih lanjut."
        }
    )

    left, right = st.columns(
        [1.05, .95],
        gap="large"
    )

    # RESULT CARD

    with left:

        st.markdown(
            "<div class='section-heading'>"
            "✨ Diagnosis Result</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class='result-card'>

                <div class='result-emoji'>
                    {info['emoji']}
                </div>

                <div class='result-kicker'>
                    PREDICTED BEARING CONDITION
                </div>

                <div class='result-name'
                style='color:{info["color"]}'>
                    {info['title']}
                </div>

                <div style='color:#526487;font-size:14px'>
                    Posisi sensor:
                    <b>{result['sensor']}</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # MAINTENANCE RECOMMENDATION

    with right:

        st.markdown(
            "<div class='section-heading'>"
            "🛠️ Maintenance Recommendation</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class='recommendation'>
                <b>{info['emoji']} Rekomendasi:</b>
                <br>
                {info['recommendation']}
            </div>

            <div class='warning-box'>
                ⚠️ Hasil prediksi merupakan alat bantu
                analisis. Konfirmasikan melalui inspeksi teknis.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ======================================
    # PROBABILITY CHART
    # ======================================

    st.markdown(
        "<div class='section-heading'>"
        "📊 Prediction Confidence</div>",
        unsafe_allow_html=True
    )

    probs = result.get("probabilities", [])

    if isinstance(probs, dict):

        probs = [
            {
                "Kondisi": key,
                "Probabilitas": value
            }
            for key, value in probs.items()
        ]

    pdf = pd.DataFrame(probs)

    if pdf.empty or not {
        "Kondisi",
        "Probabilitas"
    }.issubset(pdf.columns):

        st.warning(
            "Data probabilitas tidak tersedia."
        )
        return

    pdf["Probabilitas"] = pd.to_numeric(
        pdf["Probabilitas"],
        errors="coerce"
    ).fillna(0)

    pdf["Persentase"] = (
        pdf["Probabilitas"] * 100
    )

    pdf = pdf.sort_values(
        "Persentase",
        ascending=False
    )

    chart_col, table_col = st.columns(
        [1.7, 1],
        gap="large"
    )

    with chart_col:

        st.bar_chart(
            pdf.set_index("Kondisi")[["Persentase"]],
            horizontal=True,
            height=360,
            use_container_width=True
        )

    with table_col:

        display = pdf[
            ["Kondisi", "Persentase"]
        ].copy()

        display["Persentase"] = display[
            "Persentase"
        ].map(
            lambda x: f"{x:.2f}%"
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=280
        )

    st.caption(
        "Probabilitas menunjukkan tingkat keyakinan "
        "model terhadap masing-masing kelas, "
        "bukan jaminan kondisi fisik."
    )


# ==========================================
# DASHBOARD
# ==========================================

if page == "📊 Dashboard":

    st.markdown(
        "<div class='section-heading'>"
        "📊 System Overview</div>",
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    vals = [

        (
            "MODEL ALGORITHM",
            metadata.get(
                "model_name",
                "Extra Trees Classifier"
            )
        ),

        (
            "VALIDATION MACRO F1",
            f"{float(metadata.get('validation_macro_f1', 0) or 0)*100:.2f}%"
        ),

        (
            "TEST ACCURACY",
            f"{float(metadata.get('test_accuracy', 0) or 0)*100:.2f}%"
        )
    ]

    for col, (label, value) in zip(
        [m1, m2, m3],
        vals
    ):

        with col:

            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>
                        {label}
                    </div>

                    <div class='metric-value'>
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown(
        "<div class='section-heading'>"
        "🧩 Bearing Condition Classes</div>",
        unsafe_allow_html=True
    )

    class_cols = st.columns(4)

    class_items = [

        (
            "💚",
            "NORMAL",
            "Kondisi bearing normal."
        ),

        (
            "🔴",
            "OUTER RACE",
            "Kerusakan lintasan luar."
        ),

        (
            "🟠",
            "INNER RACE",
            "Kerusakan lintasan dalam."
        ),

        (
            "🟣",
            "BALL FAULT",
            "Kerusakan elemen bola."
        )
    ]

    for col, item in zip(
        class_cols,
        class_items
    ):

        with col:

            st.markdown(
                f"""
                <div class='class-card'>

                    <div class='class-emoji'>
                        {item[0]}
                    </div>

                    <div class='class-name'>
                        {item[1]}
                    </div>

                    <div class='class-desc'>
                        {item[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown(
        "<div class='section-heading'>"
        "📝 Recent Predictions</div>",
        unsafe_allow_html=True
    )

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(
                st.session_state.history
            ).tail(8),
            use_container_width=True
        )

    else:

        st.info(
            "Belum ada riwayat prediksi. "
            "Buka menu Diagnosis untuk memulai."
        )


# ==========================================
# DIAGNOSIS
# ==========================================

elif page == "🔧 Diagnosis":

    input_col, placeholder_col = st.columns(
        [1.05, .95],
        gap="large"
    )

    with input_col:

        st.markdown(
            "<div class='section-heading'>"
            "🔧 Input Measurement</div>",
            unsafe_allow_html=True
        )

        st.write(
            "Masukkan parameter pengukuran "
            "untuk menganalisis kondisi bearing."
        )

        with st.form("bearing_diagnosis_form"):

            posisi_sensor = st.selectbox(
                "📍 Posisi Sensor",
                ["drive_end", "fan_end"]
            )

            st.markdown(
                "#### ⚙️ Operating Parameters"
            )

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
                    step=.1
                )

                suhu_housing_C = st.number_input(
                    "Suhu Housing (°C)",
                    value=50.8,
                    step=.1
                )

                rms_g = st.number_input(
                    "RMS (g)",
                    min_value=0.0,
                    value=.4946,
                    step=.01,
                    format="%.4f"
                )

                puncak_g = st.number_input(
                    "Puncak (g)",
                    min_value=0.0,
                    value=1.4368,
                    step=.01,
                    format="%.4f"
                )

                kurtosis = st.number_input(
                    "Kurtosis",
                    value=2.209,
                    step=.01
                )

            with c2:

                crest_factor = st.number_input(
                    "Crest Factor",
                    min_value=0.0,
                    value=2.905,
                    step=.01
                )

                energi_bpfo = st.number_input(
                    "Energi BPFO",
                    min_value=0.0,
                    value=.0042,
                    step=.001,
                    format="%.5f"
                )

                energi_bpfi = st.number_input(
                    "Energi BPFI",
                    min_value=0.0,
                    value=.0834,
                    step=.001,
                    format="%.5f"
                )

                energi_bsf = st.number_input(
                    "Energi BSF",
                    min_value=0.0,
                    value=.1533,
                    step=.001,
                    format="%.5f"
                )

                frekuensi_dominan_hz = st.number_input(
                    "Frekuensi Dominan (Hz)",
                    min_value=0.0,
                    value=38.7,
                    step=.1
                )

            submitted = st.form_submit_button(
                "🔍 ANALYZE BEARING CONDITION",
                use_container_width=True
            )

    # ======================================
    # RUN PREDICTION
    # ======================================

    if submitted:

        values = {

            "posisi_sensor":
                SENSOR_MAPPING[posisi_sensor],

            "kecepatan_rpm":
                kecepatan_rpm,

            "beban_kN":
                beban_kN,

            "suhu_housing_C":
                suhu_housing_C,

            "rms_g":
                rms_g,

            "puncak_g":
                puncak_g,

            "kurtosis":
                kurtosis,

            "crest_factor":
                crest_factor,

            "energi_bpfo":
                energi_bpfo,

            "energi_bpfi":
                energi_bpfi,

            "energi_bsf":
                energi_bsf,

            "frekuensi_dominan_hz":
                frekuensi_dominan_hz
        }

        try:

            missing = [
                f for f in FEATURE_COLUMNS
                if f not in values
            ]

            if missing:

                raise ValueError(
                    f"Fitur tidak ditemukan: {missing}"
                )

            input_df = pd.DataFrame(
                [values]
            )[FEATURE_COLUMNS]

            prediction = str(
                model.predict(input_df)[0]
            )

            probabilities = model.predict_proba(
                input_df
            )[0]

            records = [

                {
                    "Kondisi": str(label),
                    "Probabilitas": float(prob)
                }

                for label, prob in zip(
                    model.classes_,
                    probabilities
                )
            ]

            result = {

                "prediction": prediction,

                "probabilities": records,

                "sensor": posisi_sensor,

                "timestamp": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

                "input": values
            }

            st.session_state.latest_result = result

            st.session_state.history.append(
                {
                    "timestamp": result["timestamp"],
                    "sensor": posisi_sensor,
                    "prediction": prediction
                }
            )

        except Exception as exc:

            st.error(
                f"Terjadi kesalahan saat prediksi: {exc}"
            )

    # ======================================
    # SHOW RESULT
    # ======================================

    if st.session_state.latest_result is None:

        with placeholder_col:

            st.markdown(
                "<div class='section-heading'>"
                "✨ Diagnosis Result</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class='result-card'
                style='padding:42px 18px'>

                    <div class='result-emoji'>
                        🔍
                    </div>

                    <h2>Waiting for Analysis</h2>

                    <p>
                        Masukkan data pengukuran,
                        lalu tekan tombol analisis
                        untuk melihat hasil diagnosis.
                    </p>

                    <div style='font-size:35px;margin-top:12px'>
                        🧒🏻 🔧 🐶
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown("---")

        show_result(
            st.session_state.latest_result
        )


# ==========================================
# PREDICTION HISTORY
# ==========================================

elif page == "📋 Riwayat Prediksi":

    st.markdown(
        "<div class='section-heading'>"
        "📋 Prediction History</div>",
        unsafe_allow_html=True
    )

    st.write(
        "Daftar hasil prediksi selama sesi aplikasi berjalan."
    )

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(
                st.session_state.history
            ),
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


# ==========================================
# MODEL INFORMATION
# ==========================================

elif page == "📈 Informasi Model":

    st.markdown(
        "<div class='section-heading'>"
        "📈 Model Information</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='panel'>

            <h3>
                🤖 {metadata.get(
                    'model_name',
                    'Extra Trees Classifier'
                )}
            </h3>

            <p>
                <b>Target:</b>
                {metadata.get('target', 'N/A')}
            </p>

            <p>
                <b>Validation Macro F1:</b>
                {float(metadata.get('validation_macro_f1', 0) or 0)*100:.2f}%
            </p>

            <p>
                <b>Test Macro F1:</b>
                {float(metadata.get('test_macro_f1', 0) or 0)*100:.2f}%
            </p>

            <p>
                <b>Test Accuracy:</b>
                {float(metadata.get('test_accuracy', 0) or 0)*100:.2f}%
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 🧩 Input Features")

    st.dataframe(
        pd.DataFrame(
            {
                "No": range(
                    1,
                    len(FEATURE_COLUMNS) + 1
                ),
                "Feature": FEATURE_COLUMNS
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### 🏷️ Classification Labels")

    st.write(
        ", ".join(
            metadata.get("classes", [])
        )
        if metadata.get("classes")
        else ", ".join(
            map(
                str,
                getattr(model, "classes_", [])
            )
        )
    )

    st.caption(
        "Model menggunakan Extra Trees Classifier "
        "yang telah dilatih menggunakan data train "
        "dan validation."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;padding:12px'>

        <div style='font-size:30px'>
            ⚙️ 🧒🏻 🐶
        </div>

        <b style='color:#26365D'>
            BearingGuard AI
        </b>

        <p style='color:#64748B;font-size:12px'>
            Machine Learning Project •
            Bearing Fault Diagnosis •
            Shinchan-inspired Edition
        </p>

    </div>
    """,
    unsafe_allow_html=True
)
