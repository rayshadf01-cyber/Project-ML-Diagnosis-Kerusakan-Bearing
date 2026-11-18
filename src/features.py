
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

SENSOR_MAPPING = {
    "drive_end": 0,
    "fan_end": 1
}


def prepare_input(data):
    """
    Menyiapkan input agar sesuai dengan urutan fitur model.
    """
    import pandas as pd

    df = pd.DataFrame([data])
    df["posisi_sensor"] = df["posisi_sensor"].map(SENSOR_MAPPING)

    return df[FEATURE_COLUMNS]
