import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# Başlık
st.title("🔧 Emniyet Kemeri Üst Bağlantı Noktası Tahmin Aracı")

st.markdown("""
Bu araç, araç içi referans noktalarına (H-Point ve kemer montaj koordinatları) göre 
emniyet kemeri üst bağlantı noktasının Z eksenindeki yüksekliğini tahmin eder.
""")

# Model eğitimi için veri dosyasını oku
@st.cache_data
def load_model():
    veri3 = pd.read_csv("veri3.csv", sep=";", decimal=",")
    X = veri3[[
        "H-Point_x",
        "H-Point_y",
        "H-Point_z",
        "Kemer Bağlantı Noktası_x",
        "Kemer Bağlantı Noktası_y"
    ]]
    y = veri3["Kemer Bağlantı Noktası_z"]

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)

    return model

# Modeli yükle
model = load_model()

st.subheader("📥 Girdi Verileri")

# Kullanıcıdan girdi al
H_Point_x = st.number_input("H-Point X (mm)", value=1350)
H_Point_y = st.number_input("H-Point Y (mm)", value=-330)
H_Point_z = st.number_input("H-Point Z (mm)", value=400)
Kemer_Baglanti_x = st.number_input("Kemer Bağlantı Noktası X (mm)", value=1560)
Kemer_Baglanti_y = st.number_input("Kemer Bağlantı Noktası Y (mm)", value=-600)

# Tahmin butonu
if st.button("🔍 Tahmin Et"):
    input_df = pd.DataFrame([{
        "H-Point_x": H_Point_x,
        "H-Point_y": H_Point_y,
        "H-Point_z": H_Point_z,
        "Kemer Bağlantı Noktası_x": Kemer_Baglanti_x,
        "Kemer Bağlantı Noktası_y": Kemer_Baglanti_y
    }])

    predicted_z = model.predict(input_df)[0]
    delta_z = predicted_z - H_Point_z

    st.success(f"🎯 Tahmini Kemer Üst Bağlantı Noktası Z: **{predicted_z:.2f} mm**")
    st.info(f"📏 Delta Z (Kemer Z - H-point Z): **{delta_z:.2f} mm**")

    if delta_z < 0:
        st.warning("⚠️ Kemer bağlantı noktası H-point'in altında.")
    elif delta_z > 0:
        st.success("✅ Kemer bağlantı noktası H-point'in üstünde.")
    else:
        st.info("➖ Kemer bağlantı noktası ile H-point aynı seviyede.")
