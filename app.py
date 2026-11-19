
import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="BearingGuard AI",
    page_icon="🔴",
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
# SHINCHAN-INSPIRED CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* MAIN THEME */
.stApp {
    background: #FFF8EA;
    color: #26345C;
}

/* HIDE DEFAULT STREAMLIT ELEMENTS */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: visible;}

/* MAIN CONTENT */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #26345C;
    border-right: 4px solid #F4C542;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

section[data-testid="stSidebar"] .stRadio label {
    background: #344775;
    border-radius: 14px;
    padding: 12px 15px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: #D9363E;
    transform: translateX(4px);
}

/* HEADINGS */
h1, h2, h3 {
    color: #26345C !important;
    font-weight: 800 !important;
}

/* HEADER CARD */
.hero-card {
    background: linear-gradient(120deg, #D9363E, #B92735);
    padding: 28px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 22px rgba(130, 30, 40, 0.15);
}

.hero-card h1 {
    color: white !important;
    margin-bottom: 5px;
}

.hero-card p {
    color: #FFF4D9;
    margin-bottom: 0;
}

/* CARDS */
.custom-card {
    background: #FFFFFF;
    border: 2px solid #F1DFC8;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(38, 52, 92, 0.05);
}

/* METRIC CARDS */
.metric-card {
    background: #FFFFFF;
    border: 2px solid #F1DFC8;
    border-radius: 16px;
    padding: 18px;
    min-height: 105px;
}

.metric-label {
    color: #777777;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.metric-value {
    color: #D9363E;
    font-size: 25px;
    font-weight: 900;
    margin-top: 8px;
}

/* INPUTS */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px !important;
    border-color: #E8D5C0 !important;
    background: #FFFCF7 !important;
}

div[data-baseweb="input"] input {
    color: #26345C !important;
}

/* BUTTONS */
.stButton button,
.stFormSubmitButton button {
    background: #D9363E !important;
    color: white !important;
    border: none !important;
    border-radius: 13px !important;
    font-weight: 800 !important;
    min-height: 46px;
    transition: all 0.2s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #F4C542 !important;
    color: #26345C !important;
    transform: translateY(-2px);
}

/* RESULT */
.result-card {
    background: #FFFFFF;
    border: 3px solid #F4C542;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 16px;
}

.result-label {
    font-size: 12px;
    color: #777777;
    font-weight: 800;
    text-transform: uppercase;
}

.result-value {
    font-size: 32px;
    font-weight: 900;
    margin-top: 8px;
}

/* INFO BOX */
.info-card {
    background: #FFF0C2;
    border-left: 6px solid #F4C542;
    padding: 15px;
    border-radius: 12px;
    color: #26345C;
    margin-top: 12px;
}

/* MOBILE */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem 0.8rem;
    }

    .hero-card {
        padding: 20px;
    }

    .hero-card h1 {
        font-size: 25px;
    }

    .metric-value {
        font-size: 20px;
    }

    .result-value {
        font-size: 26px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()
except Exception as error:
    st.error("Model gagal dimuat. Periksa file model dan metadata.")
    st.exception(error)
    st.stop()


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =====================================================
# RECOMMENDATIONS
# =====================================================

RECOMMENDATIONS = {
    "normal": {
        "title": "Bearing dalam kondisi normal",
        "description": (
            "Tidak ditemukan indikasi kerusakan dominan "
            "berdasarkan hasil klasifikasi model."
        ),
        "action": (
            "Lanjutkan pemantauan kondisi dan perawatan "
            "berkala sesuai jadwal."
        ),
        "color": "#3C8D68"
    },

    "outer_race": {
        "title": "Indikasi kerusakan Outer Race",
        "description": (
            "Pola pengukuran mengindikasikan kemungkinan "
            "kerusakan pada bagian outer race."
        ),
        "action": (
            "Lakukan inspeksi bearing, periksa getaran, "
            "dan pertimbangkan penggantian komponen."
        ),
        "color": "#D9363E"
    },

    "inner_race": {
        "title": "Indikasi kerusakan Inner Race",
        "description": (
            "Pola pengukuran mengindikasikan kemungkinan "
            "kerusakan pada bagian inner race."
        ),
        "action": (
            "Jadwalkan pemeriksaan lebih lanjut dan "
            "siapkan komponen pengganti."
        ),
        "color": "#A84B70"
    },

    "bola": {
        "title": "Indikasi kerusakan elemen bola",
        "description": (
            "Pola pengukuran mengindikasikan kemungkinan "
            "kerusakan pada elemen bola bearing."
        ),
        "action": (
            "Periksa kondisi elemen bola dan siapkan "
            "komponen pengganti jika diperlukan."
        ),
        "color": "#C47B25"
    }
}


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#F4C542;'>⚙️ BearingGuard</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#F8EBD5;'>SMART BEARING DIAGNOSTIC SYSTEM</p>",
        unsafe_allow_html=True
    )

    st.divider()

    menu = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "🔧 Diagnosis",
            "📋 Riwayat Prediksi",
            "📊 Informasi Model"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(
        """
        <div style='background:#344775;padding:15px;border-radius:15px;'>
            <h4 style='color:#F4C542;'>🌟 Shinchan Mode</h4>
            <p style='font-size:12px;color:white;'>
                Keep your machines happy and healthy!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# HEADER FUNCTION
# =====================================================

def show_header(title, subtitle):

    st.markdown(
        f"""
        <div class="hero-card">
            <h1>🔴 {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# DASHBOARD PAGE
# =====================================================

if menu == "🏠 Dashboard":

    show_header(
        "BearingGuard AI",
        "Intelligent Bearing Fault Diagnosis System"
    )

    st.markdown("## 🌟 System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Algorithm</div>
                <div class="metric-value">Extra Trees</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        val_f1 = metadata.get("test_macro_f1", 0) * 100

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Macro F1</div>
                <div class="metric-value">{val_f1:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        accuracy = metadata.get("test_accuracy", 0) * 100

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">{accuracy:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown(
            """
            <div class="custom-card">
                <h3>🔧 About This System</h3>
                <p>
                BearingGuard AI merupakan aplikasi klasifikasi
                kondisi bearing menggunakan algoritma
                machine learning Extra Trees.
                </p>
                <p>
                Sistem menerima parameter pengukuran getaran
                dan menghasilkan prediksi kondisi bearing.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="custom-card">
                <h3>🎯 Diagnosis Classes</h3>
                <p>🟢 Normal</p>
                <p>🔴 Outer Race</p>
                <p>🌸 Inner Race</p>
                <p>🟡 Bola</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Pilih menu Diagnosis di sidebar untuk mulai "
        "melakukan klasifikasi kondisi bearing."
    )


# =====================================================
# DIAGNOSIS PAGE
# =====================================================

elif menu == "🔧 Diagnosis":

    show_header(
        "Bearing Diagnosis",
        "Masukkan parameter pengukuran untuk menganalisis kondisi bearing."
    )

    input_col, result_col = st.columns(
        [1.15, 0.85],
        gap="large"
    )

    with input_col:

        st.markdown("## 🔧 Input Measurement")

        with st.form("bearing_form"):

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
                    step=10.0
                )

                beban_kN = st.number_input(
                    "Beban (kN)",
                    min_value=0.0,
                    value=2.3
                )

                suhu_housing_C = st.number_input(
                    "Suhu Housing (°C)",
                    value=50.8
                )

            with c2:

                rms_g = st.number_input(
                    "RMS (g)",
                    min_value=0.0,
                    value=0.4946
                )

                puncak_g = st.number_input(
                    "Puncak (g)",
                    min_value=0.0,
                    value=1.4368
                )

                frekuensi_dominan_hz = st.number_input(
                    "Frekuensi Dominan (Hz)",
                    min_value=0.0,
                    value=38.7
                )

            st.markdown("### 📈 Vibration Characteristics")

            c3, c4 = st.columns(2)

            with c3:

                kurtosis = st.number_input(
                    "Kurtosis",
                    value=2.209
                )

                crest_factor = st.number_input(
                    "Crest Factor",
                    min_value=0.0,
                    value=2.905
                )

                energi_bpfo = st.number_input(
                    "Energi BPFO",
                    min_value=0.0,
                    value=0.0042,
                    format="%.5f"
                )

            with c4:

                energi_bpfi = st.number_input(
                    "Energi BPFI",
                    min_value=0.0,
                    value=0.0834,
                    format="%.5f"
                )

                energi_bsf = st.number_input(
                    "Energi BSF",
                    min_value=0.0,
                    value=0.1533,
                    format="%.5f"
                )

            st.write("")

            submitted = st.form_submit_button(
                "🔴 ANALYZE BEARING CONDITION",
                use_container_width=True
            )

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------

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

            input_data = pd.DataFrame([input_values])

            # Follow the exact feature order used during training
            input_data = input_data[metadata["features"]]

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
                "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Sensor": posisi_sensor,
                "Prediksi": prediction,
                "Confidence": float(max(probabilities) * 100)
            }

            st.session_state.last_result = {
                "prediction": prediction,
                "probability_df": probability_df,
                "input": input_values
            }

            st.session_state.history.insert(0, result)

        except Exception as error:
            st.error("Prediksi gagal dilakukan.")
            st.exception(error)

    # -------------------------------------------------
    # RESULT PANEL
    # -------------------------------------------------

    with result_col:

        st.markdown("## ✨ Diagnosis Result")

        if st.session_state.last_result is None:

            st.markdown(
                """
                <div class="custom-card" style="text-align:center;">
                    <h1>🔍</h1>
                    <h3>Waiting for Analysis</h3>
                    <p>Masukkan data pengukuran dan tekan tombol
                    analisis untuk melihat hasil diagnosis.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            result = st.session_state.last_result

            prediction = result["prediction"]
            probability_df = result["probability_df"]

            recommendation = RECOMMENDATIONS.get(
                prediction,
                RECOMMENDATIONS["normal"]
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        PREDICTED BEARING CONDITION
                    </div>
                    <div class="result-value"
                         style="color:{recommendation['color']};">
                        {prediction.upper()}
                    </div>
                    <p>{recommendation['title']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📊 Prediction Confidence")

            chart_data = probability_df.set_index(
                "Kondisi"
            )["Persentase"]

            st.bar_chart(
                chart_data,
                color="#D9363E"
            )

            st.markdown("### 📋 Probability Details")

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

            st.markdown("### 🛠️ Maintenance Recommendation")

            st.info(recommendation["description"])

            st.markdown(
                f"""
                <div class="info-card">
                    <b>Recommended Action</b><br>
                    {recommendation['action']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "Hasil prediksi merupakan alat bantu analisis "
                "dan perlu dikonfirmasi melalui inspeksi teknis."
            )


# =====================================================
# HISTORY PAGE
# =====================================================

elif menu == "📋 Riwayat Prediksi":

    show_header(
        "Prediction History",
        "Riwayat diagnosis yang dilakukan selama sesi aplikasi ini."
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

        if st.button("🗑️ Hapus Riwayat"):

            st.session_state.history = []
            st.session_state.last_result = None

            st.rerun()

    else:

        st.info(
            "Belum ada riwayat prediksi. "
            "Silakan lakukan diagnosis terlebih dahulu."
        )


# =====================================================
# MODEL INFORMATION PAGE
# =====================================================

elif menu == "📊 Informasi Model":

    show_header(
        "Model Information",
        "Informasi model machine learning yang digunakan."
    )

    st.markdown("## 🤖 Model Overview")

    st.markdown(
        """
        <div class="custom-card">
            <h3>Extra Trees Classifier</h3>
            <p>
            Model digunakan untuk mengklasifikasikan kondisi
            bearing berdasarkan parameter pengukuran getaran.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Validation Macro F1",
            f"{metadata.get('validation_macro_f1', 0):.4f}"
        )

    with c2:
        st.metric(
            "Test Macro F1",
            f"{metadata.get('test_macro_f1', 0):.4f}"
        )

    with c3:
        st.metric(
            "Test Accuracy",
            f"{metadata.get('test_accuracy', 0):.4f}"
        )

    st.markdown("### 🧩 Model Features")

    st.dataframe(
        pd.DataFrame({
            "Feature": metadata["features"]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🎯 Classification Classes")

    st.write(metadata["classes"])


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;padding:15px;color:#777;">
        <b>🔴 BearingGuard AI</b><br>
        Smart Bearing Fault Diagnosis System<br>
        <small>Machine Learning Project · P4–P5</small>
    </div>
    """,
    unsafe_allow_html=True
)
