
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

# =====================================================
# 2. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()


# =====================================================
# 3. SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "active_page" not in st.session_state:
    st.session_state.active_page = "Diagnosis"


# =====================================================
# 4. COLOR AND CLASS CONFIGURATION
# =====================================================

NAVY = "#26345C"
RED = "#D8323F"
YELLOW = "#FFD45A"
CREAM = "#FFF8EA"
WHITE = "#FFFFFF"
DARK = "#293044"

CLASS_INFO = {
    "normal": {
        "label": "NORMAL",
        "emoji": "🟢",
        "color": "#218C58",
        "description": "Bearing berada dalam kondisi normal."
    },
    "outer_race": {
        "label": "OUTER RACE",
        "emoji": "🔴",
        "color": "#D8323F",
        "description": "Terindikasi kerusakan pada bagian outer race."
    },
    "inner_race": {
        "label": "INNER RACE",
        "emoji": "🟠",
        "color": "#E58B27",
        "description": "Terindikasi kerusakan pada bagian inner race."
    },
    "bola": {
        "label": "BALL FAULT",
        "emoji": "🟡",
        "color": "#B78A00",
        "description": "Terindikasi kerusakan pada elemen bola bearing."
    }
}

RECOMMENDATIONS = {
    "normal": (
        "Bearing terdeteksi dalam kondisi normal. "
        "Lanjutkan pemantauan getaran dan perawatan berkala."
    ),
    "outer_race": (
        "Terindikasi kerusakan outer race. "
        "Lakukan inspeksi pada lintasan luar bearing dan "
        "pertimbangkan penggantian apabila kerusakan terkonfirmasi."
    ),
    "inner_race": (
        "Terindikasi kerusakan inner race. "
        "Jadwalkan pemeriksaan lebih lanjut dan siapkan "
        "komponen bearing pengganti."
    ),
    "bola": (
        "Terindikasi kerusakan elemen bola. "
        "Periksa kondisi elemen gelinding dan siapkan "
        "komponen pengganti apabila diperlukan."
    )
}


# =====================================================
# 5. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* GENERAL */
.stApp {
    background-color: #FFF8EA;
    color: #293044;
    font-family: 'Trebuchet MS', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(255,248,234,0.95);
}

[data-testid="stAppViewContainer"] {
    background-color: #FFF8EA;
}

/* MAIN CONTENT */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #26345C;
    border-right: 4px solid #FFD45A;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFD45A !important;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #26345C !important;
    font-weight: 800 !important;
}

p, li, label {
    color: #35415F;
}

/* INPUT LABELS */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stNumberInput label {
    color: #26345C !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* INPUT FIELDS */
div[data-baseweb="input"] input {
    background-color: #FFFCF7 !important;
    color: #26345C !important;
    -webkit-text-fill-color: #26345C !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFCF7 !important;
    border: 1px solid #D8BFA8 !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] * {
    color: #26345C !important;
}

/* BUTTON */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #D8323F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    padding: 0.65rem 1rem !important;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #B92331 !important;
    color: #FFFFFF !important;
    transform: translateY(-2px);
}

/* CARDS */
.custom-card {
    background-color: #FFFFFF;
    border: 1px solid #E8D7C3;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow: 0 5px 16px rgba(38,52,92,0.07);
}

.hero-card {
    background: linear-gradient(120deg, #26345C, #354B7D);
    border-radius: 20px;
    padding: 28px;
    color: white !important;
    border-bottom: 5px solid #FFD45A;
    margin-bottom: 24px;
}

.hero-card h1,
.hero-card h2,
.hero-card p {
    color: white !important;
}

.metric-card {
    background-color: white;
    border: 1px solid #E8D7C3;
    border-radius: 15px;
    padding: 17px;
    box-shadow: 0 3px 12px rgba(38,52,92,0.05);
}

.metric-label {
    color: #667085;
    font-size: 13px;
    font-weight: 700;
}

.metric-value {
    color: #26345C;
    font-size: 24px;
    font-weight: 900;
    margin-top: 5px;
}

/* RESULT CARD */
.result-card {
    background-color: #FFFFFF;
    border-radius: 18px;
    border: 2px solid #E8D7C3;
    padding: 25px;
    margin-bottom: 18px;
    box-shadow: 0 5px 16px rgba(38,52,92,0.07);
}

.result-label {
    font-size: 13px;
    color: #667085;
    font-weight: 800;
    letter-spacing: 1px;
}

.result-value {
    font-size: 30px;
    font-weight: 900;
    margin-top: 8px;
}

/* BADGE */
.badge {
    display: inline-block;
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 800;
}

/* DIVIDER */
hr {
    border-color: #E8D7C3 !important;
}

/* EXPANDER */
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #E8D7C3;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #26345C !important;
    font-weight: 700 !important;
}

/* MOBILE RESPONSIVE */
@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem 0.8rem;
    }

    .hero-card {
        padding: 20px;
    }

    .result-value {
        font-size: 24px;
    }

    .metric-value {
        font-size: 20px;
    }

    h1 {
        font-size: 27px !important;
    }

    h2 {
        font-size: 22px !important;
    }

    [data-testid="stSidebar"] {
        min-width: 0;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 6. HELPER FUNCTIONS
# =====================================================

def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_result(result):
    prediction = result["prediction"]
    info = CLASS_INFO.get(prediction, CLASS_INFO["normal"])
    probabilities = result["probabilities"]

    st.markdown("### ✨ Diagnosis Result")

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">PREDICTED BEARING CONDITION</div>
        <div class="result-value" style="color:{info['color']}">
            {info['emoji']} {info['label']}
        </div>
        <p style="color:#35415F;margin-top:10px;">
            {info['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Prediction Confidence")

    probability_df = pd.DataFrame({
        "Kondisi": list(probabilities.keys()),
        "Probabilitas (%)": [
            value * 100 for value in probabilities.values()
        ]
    }).sort_values("Probabilitas (%)", ascending=True)

    st.bar_chart(
        probability_df.set_index("Kondisi"),
        horizontal=True
    )

    st.markdown("### 🛠️ Maintenance Recommendation")

    st.info(RECOMMENDATIONS.get(
        prediction,
        "Lakukan pemeriksaan teknis lebih lanjut."
    ))

    st.warning(
        "Lakukan inspeksi teknis lanjutan sebelum menentukan "
        "tindakan perbaikan atau penggantian bearing."
    )


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 20px 5px;">
        <div style="font-size:35px;">⚙️</div>
        <h2 style="color:#FFD45A !important;margin-bottom:2px;">
            BearingGuard
        </h2>
        <p style="font-size:12px;color:#FFFFFF !important;">
            SMART BEARING DIAGNOSTIC SYSTEM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### MAIN MENU")

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔧 Diagnosis",
            "📋 Riwayat Prediksi",
            "📊 Informasi Model"
        ],
        label_visibility="collapsed",
        index=[
            "🏠 Dashboard",
            "🔧 Diagnosis",
            "📋 Riwayat Prediksi",
            "📊 Informasi Model"
        ].index(
            {
                "Dashboard": "🏠 Dashboard",
                "Diagnosis": "🔧 Diagnosis",
                "Riwayat Prediksi": "📋 Riwayat Prediksi",
                "Informasi Model": "📊 Informasi Model"
            }[st.session_state.active_page]
        )
    )

    page_mapping = {
        "🏠 Dashboard": "Dashboard",
        "🔧 Diagnosis": "Diagnosis",
        "📋 Riwayat Prediksi": "Riwayat Prediksi",
        "📊 Informasi Model": "Informasi Model"
    }

    st.session_state.active_page = page_mapping[selected_page]

    st.markdown("---")

    st.markdown("""
    <div style="
        background:#354B7D;
        border:1px solid #526A9B;
        border-radius:15px;
        padding:15px;
        margin-top:15px;">
        <h3 style="color:#FFD45A !important;font-size:16px;">
            🌟 Shinchan Mode
        </h3>
        <p style="font-size:12px;color:white !important;">
            Keep your machines happy and healthy!
        </p>
        <p style="font-size:22px;">🌻 ⚙️ ❤️</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("BearingGuard AI | Machine Learning Project")


# =====================================================
# 8. HERO HEADER
# =====================================================

st.markdown("""
<div class="hero-card">
    <div style="
        font-size:11px;
        font-weight:800;
        letter-spacing:2px;
        color:#FFD45A;">
        INDUSTRIAL AI • PREDICTIVE MAINTENANCE
    </div>
    <h1 style="margin-top:10px;">
        ⚙️ BearingGuard AI
    </h1>
    <p>
        Intelligent Bearing Fault Diagnosis System
    </p>
    <p style="font-size:13px;">
        Platform klasifikasi kondisi bearing berdasarkan
        parameter pengukuran getaran mesin.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 9. DASHBOARD PAGE
# =====================================================

if st.session_state.active_page == "Dashboard":

    st.title("🏠 Dashboard")

    st.write(
        "Selamat datang di BearingGuard AI. "
        "Pantau kondisi bearing dan lakukan diagnosis berbasis machine learning."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "MODEL ALGORITHM",
            metadata.get("model_name", "Extra Trees")
        )

    with col2:
        metric_card(
            "VALIDATION MACRO F1",
            f"{metadata.get('validation_macro_f1', 0)*100:.2f}%"
        )

    with col3:
        metric_card(
            "TEST ACCURACY",
            f"{metadata.get('test_accuracy', 0)*100:.2f}%"
        )

    st.markdown("### 🔍 Bearing Condition Classes")

    class_cols = st.columns(4)

    for column, (key, info) in zip(class_cols, CLASS_INFO.items()):
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:27px;">{info['emoji']}</div>
                <div style="color:{info['color']};font-weight:900;">
                    {info['label']}
                </div>
                <div style="font-size:12px;color:#667085;">
                    {info['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Start Diagnosis")

    st.write(
        "Masuk ke menu Diagnosis untuk memasukkan parameter "
        "pengukuran dan mendapatkan hasil prediksi kondisi bearing."
    )

    if st.button("🔧 Mulai Diagnosis", use_container_width=True):
        st.session_state.active_page = "Diagnosis"
        st.rerun()


# =====================================================
# 10. DIAGNOSIS PAGE
# =====================================================

elif st.session_state.active_page == "Diagnosis":

    st.title("🔧 Bearing Diagnosis")

    st.write(
        "Masukkan parameter pengukuran bearing untuk "
        "menganalisis kemungkinan kerusakan."
    )

    input_col, result_col = st.columns(
        [1.25, 0.95],
        gap="large"
    )

    # ---------------- INPUT LEFT ----------------

    with input_col:

        st.markdown("### 🛠️ Input Measurement")

        with st.form("bearing_form"):

            posisi_sensor = st.selectbox(
                "📍 Posisi Sensor",
                ["drive_end", "fan_end"]
            )

            st.markdown("#### ⚙️ Operating Parameters")

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
                    value=50.80,
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

            with c2:
                frekuensi_dominan_hz = st.number_input(
                    "Frekuensi Dominan (Hz)",
                    min_value=0.0,
                    value=38.70,
                    step=0.1
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

            st.markdown("#### 📈 Vibration Characteristics")

            c3, c4 = st.columns(2)

            with c3:
                kurtosis = st.number_input(
                    "Kurtosis",
                    value=2.209,
                    step=0.1
                )

            with c4:
                crest_factor = st.number_input(
                    "Crest Factor",
                    min_value=0.0,
                    value=2.905,
                    step=0.1
                )

            st.markdown("---")

            submitted = st.form_submit_button(
                "✨ ANALYZE BEARING CONDITION",
                use_container_width=True
            )

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

            try:
                input_data = input_data[metadata["features"]]

                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]

                probability_dict = {
                    str(label): float(probability)
                    for label, probability in zip(
                        model.classes_,
                        probabilities
                    )
                }

                result = {
                    "prediction": str(prediction),
                    "probabilities": probability_dict,
                    "sensor": posisi_sensor,
                    "time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    )
                }

                st.session_state.latest_result = result
                st.session_state.history.insert(0, result)

            except Exception as error:
                st.error(f"Prediksi gagal: {error}")

    # ---------------- RESULT RIGHT ----------------

    with result_col:

        if st.session_state.latest_result is not None:

            render_result(st.session_state.latest_result)

            st.caption(
                "Hasil prediksi merupakan alat bantu analisis "
                "dan perlu dikonfirmasi melalui inspeksi teknis."
            )

        else:

            st.markdown("### ✨ Diagnosis Result")

            st.markdown("""
            <div class="result-card"
                style="text-align:center;padding:45px 20px;">
                <div style="font-size:55px;">🔎</div>
                <h2 style="color:#26345C !important;">
                    Waiting for Analysis
                </h2>
                <p style="color:#35415F;">
                    Masukkan data pengukuran dan tekan tombol
                    analisis untuk melihat hasil diagnosis.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 💡 How it works")

            st.markdown("""
            <div class="custom-card">
                <b style="color:#26345C;">01. Input Measurement</b>
                <p>Masukkan parameter getaran bearing.</p>
                <b style="color:#26345C;">02. AI Analysis</b>
                <p>Model Extra Trees menganalisis data pengukuran.</p>
                <b style="color:#26345C;">03. Diagnosis Result</b>
                <p>Lihat kelas kerusakan dan rekomendasi pemeliharaan.</p>
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# 11. HISTORY PAGE
# =====================================================

elif st.session_state.active_page == "Riwayat Prediksi":

    st.title("📋 Riwayat Prediksi")

    st.write(
        "Daftar hasil diagnosis yang dilakukan selama sesi aplikasi ini."
    )

    if len(st.session_state.history) > 0:

        history_data = []

        for item in st.session_state.history:
            history_data.append({
                "Waktu": item["time"],
                "Posisi Sensor": item["sensor"],
                "Prediksi": item["prediction"].upper(),
                "Confidence": (
                    max(item["probabilities"].values()) * 100
                )
            })

        history_df = pd.DataFrame(history_data)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Riwayat CSV",
            data=csv,
            file_name="riwayat_diagnosis_bearing.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.session_state.latest_result = None
            st.rerun()

    else:

        st.info(
            "Belum ada riwayat prediksi. "
            "Silakan lakukan diagnosis terlebih dahulu."
        )


# =====================================================
# 12. MODEL INFORMATION PAGE
# =====================================================

elif st.session_state.active_page == "Informasi Model":

    st.title("📊 Informasi Model")

    st.write(
        "Informasi model machine learning yang digunakan "
        "untuk klasifikasi kondisi bearing."
    )

    st.markdown("### 🤖 Model Overview")

    col1, col2 = st.columns(2)

    with col1:
        metric_card(
            "MODEL NAME",
            metadata.get("model_name", "Extra Trees")
        )

    with col2:
        metric_card(
            "MODEL TYPE",
            metadata.get("model_type", "ExtraTreesClassifier")
        )

    st.markdown("### 📈 Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "VALIDATION MACRO F1",
            f"{metadata.get('validation_macro_f1', 0)*100:.2f}%"
        )

    with c2:
        metric_card(
            "TEST MACRO F1",
            f"{metadata.get('test_macro_f1', 0)*100:.2f}%"
        )

    with c3:
        metric_card(
            "TEST ACCURACY",
            f"{metadata.get('test_accuracy', 0)*100:.2f}%"
        )

    st.markdown("### 🎯 Classification Target")

    target_classes = pd.DataFrame({
        "Class": list(CLASS_INFO.keys()),
        "Description": [
            info["description"]
            for info in CLASS_INFO.values()
        ]
    })

    st.dataframe(
        target_classes,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🧩 Features Used")

    features = metadata.get("features", [])

    feature_cols = st.columns(3)

    for index, feature in enumerate(features):
        with feature_cols[index % 3]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:10px;">
                <span style="color:#26345C;font-weight:700;">
                    {feature}
                </span>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Detail Konfigurasi Model"):
        st.json(metadata.get("best_params", {}))


# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center;padding:15px;">
    <p style="color:#26345C;font-weight:800;">
        ⚙️ BearingGuard AI
    </p>
    <p style="color:#667085;font-size:12px;">
        Machine Learning Project • Bearing Fault Diagnosis
    </p>
    <p style="color:#667085;font-size:11px;">
        Prediction results are decision-support information
        and must be verified through technical inspection.
    </p>
    <p style="font-size:20px;">🌻 ❤️ ⚙️</p>
</div>
""", unsafe_allow_html=True)
