import streamlit as st
import pandas as pd
from model_utils import load_model

st.set_page_config(
    page_title="Прогноз недвижимости",
    page_icon="🏙️",
    layout="centered"
)

@st.cache_resource
def load_cached_model():
    return load_model()

@st.cache_data
def load_data():
    df = pd.read_csv("realty_data.csv")
    df['rooms'] = df['rooms'].fillna(df['rooms'].median())
    df['city'] = df['city'].fillna('—')
    df['district'] = df['district'].fillna('—')
    df['lat'] = df['lat'].fillna(df['lat'].mean())
    df['lon'] = df['lon'].fillna(df['lon'].mean())
    return df

def run_app(regressor, data_df):
    st.title("🏠 Прогноз стоимости недвижимости")

    city_options = ['—'] + sorted(data_df['city'].dropna().unique().tolist())
    selected_city = st.selectbox('Город', city_options)

    selected_district = '—'
    if selected_city != '—':
        filtered_df = data_df[data_df['city'] == selected_city]
        available_districts = filtered_df['district'].dropna().unique().tolist()

        if available_districts:
            district_options = ['—'] + sorted(available_districts)
            selected_district = st.selectbox('Район', district_options)
        else:
            st.info("Для выбранного города районы не указаны. Продолжайте без выбора района.")

    total_square = st.number_input('Общая площадь (м²)', min_value=10.0, max_value=500.0, step=1.0)
    rooms = st.number_input('Количество комнат', min_value=1, max_value=10, step=1)
    floor = st.number_input('Этаж', min_value=1, max_value=50, step=1)

    if selected_city != '—':
        coord_df = data_df[data_df['city'] == selected_city]
        if selected_district != '—':
            coord_df = coord_df[coord_df['district'] == selected_district]

        lat = coord_df['lat'].mean()
        lon = coord_df['lon'].mean()
    else:
        lat = data_df['lat'].mean()
        lon = data_df['lon'].mean()

    if selected_city != '—':
        input_df = pd.DataFrame({
            'total_square': [total_square],
            'rooms': [rooms],
            'floor': [floor],
            'lat': [lat],
            'lon': [lon],
            'city': [selected_city],
            'district': [selected_district]
        })

        if st.button('Прогнозировать'):
            prediction = regressor.predict(input_df)
            st.success(f"💰 Прогнозируемая цена: {prediction[0]:,.2f} руб.")
    else:
        st.warning("Пожалуйста, выберите город для прогноза.")

model = load_cached_model()
data = load_data()
run_app(model, data)