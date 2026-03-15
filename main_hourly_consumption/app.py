import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Energy Consumption Simulator",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FULL_MODEL_PATH = os.path.join(RESULTS_DIR, "lightgbm_full.pkl")
NO_LAGS_MODEL_PATH = os.path.join(RESULTS_DIR, "lightgbm_no_lags.pkl")


@st.cache_resource
def load_models():
    full_model = None
    no_lags_model = None

    if os.path.exists(FULL_MODEL_PATH):
        full_model = joblib.load(FULL_MODEL_PATH)

    if os.path.exists(NO_LAGS_MODEL_PATH):
        no_lags_model = joblib.load(NO_LAGS_MODEL_PATH)

    return full_model, no_lags_model


full_model, no_lags_model = load_models()

BASE_FEATURES = [
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "Chicago_temp",
    "Chicago_humidity",
    "Chicago_pressure",
    "Chicago_wind_speed",
]

FULL_FEATURES = BASE_FEATURES + [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_max_12",
    "rolling_min_12",
]

def build_input_df(
    hour,
    weekday,
    month,
    is_holiday,
    temp,
    humidity,
    pressure,
    wind_speed,
    lag_values=None,
):
    is_weekend = 1 if weekday >= 5 else 0

    row = {
        "hour": hour,
        "weekday": weekday,
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "Chicago_temp": temp,
        "Chicago_humidity": humidity,
        "Chicago_pressure": pressure,
        "Chicago_wind_speed": wind_speed,
    }

    if lag_values is not None:
        row.update(lag_values)

    return pd.DataFrame([row])


def prepare_full_lag_values(lag_1, lag_2, lag_3, lag_6, lag_12, lag_24):
    rolling_mean_3 = np.mean([lag_1, lag_2, lag_3])
    rolling_mean_6 = np.mean([lag_1, lag_2, lag_3, lag_6, lag_12, lag_24])
    rolling_max_12 = np.max([lag_1, lag_2, lag_3, lag_6, lag_12])
    rolling_min_12 = np.min([lag_1, lag_2, lag_3, lag_6, lag_12])

    return {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_6": lag_6,
        "lag_12": lag_12,
        "lag_24": lag_24,
        "rolling_mean_3": rolling_mean_3,
        "rolling_mean_6": rolling_mean_6,
        "rolling_max_12": rolling_max_12,
        "rolling_min_12": rolling_min_12,
    }


def predict_consumption(model, df, feature_cols):
    return float(model.predict(df[feature_cols])[0])


def make_temperature_sensitivity_plot(model, feature_cols, current_params, lag_values=None):
    temps = np.linspace(-20, 40, 61)
    preds = []

    for t in temps:
        df = build_input_df(
            hour=current_params["hour"],
            weekday=current_params["weekday"],
            month=current_params["month"],
            is_holiday=current_params["is_holiday"],
            temp=float(t),
            humidity=current_params["humidity"],
            pressure=current_params["pressure"],
            wind_speed=current_params["wind_speed"],
            lag_values=lag_values,
        )
        preds.append(predict_consumption(model, df, feature_cols))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(temps, preds)
    ax.set_title("Чутливість прогнозу до температури")
    ax.set_xlabel("Температура")
    ax.set_ylabel("Прогнозоване споживання, MW")
    ax.grid(True)
    plt.tight_layout()
    return fig


def make_hourly_profile_plot(model, feature_cols, current_params, lag_values=None):
    hours = list(range(24))
    preds = []

    for h in hours:
        df = build_input_df(
            hour=h,
            weekday=current_params["weekday"],
            month=current_params["month"],
            is_holiday=current_params["is_holiday"],
            temp=current_params["temp"],
            humidity=current_params["humidity"],
            pressure=current_params["pressure"],
            wind_speed=current_params["wind_speed"],
            lag_values=lag_values,
        )
        preds.append(predict_consumption(model, df, feature_cols))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, preds, marker="o")
    ax.set_title("Добовий профіль споживання")
    ax.set_xlabel("Година")
    ax.set_ylabel("Прогнозоване споживання, MW")
    ax.grid(True)
    plt.tight_layout()
    return fig

st.title("ENERGY CONSUMPTION PREDICTOR")
st.caption("Прогнозування енергоспоживання будівель з використанням машинного навчання")


if full_model is None and no_lags_model is None:
    st.error(
        "Не знайдено жодної моделі. Збережи хоча б одну модель у папку results "
        "під назвами lightgbm_full.pkl або lightgbm_no_lags.pkl"
    )
    st.stop()

mode_options = []
if no_lags_model is not None:
    mode_options.append("Інтерпретаційний режим (без lag/rolling ознак)")
if full_model is not None:
    mode_options.append("Прогнозний режим (з lag/rolling ознаками)")

mode = st.radio("Оберіть режим роботи", mode_options)


with st.sidebar:
    st.header("Вхідні параметри")

    preset = st.selectbox(
        "Швидкий сценарій",
        [
            "Базовий день",
            "Спекотний день",
            "Холодний день",
            "Вихідний день",
            "Святковий день",
        ]
    )

    # Значення за замовчуванням
    default_hour = 12
    default_weekday = 2
    default_month = 7
    default_holiday = 0
    default_temp = 20.0
    default_humidity = 55.0
    default_pressure = 1015.0
    default_wind = 3.0

    if preset == "Спекотний день":
        default_temp = 34.0
        default_month = 7
    elif preset == "Холодний день":
        default_temp = -10.0
        default_month = 1
    elif preset == "Вихідний день":
        default_weekday = 6
    elif preset == "Святковий день":
        default_holiday = 1

    hour = st.slider("Година", 0, 23, default_hour)
    weekday = st.slider("День тижня (0=Пн, 6=Нд)", 0, 6, default_weekday)
    month = st.slider("Місяць", 1, 12, default_month)
    is_holiday = st.selectbox("Свято", [0, 1], index=default_holiday)

    temp = st.slider("Температура, °C", -30.0, 45.0, float(default_temp), step=0.5)
    humidity = st.slider("Вологість, %", 0.0, 100.0, float(default_humidity), step=1.0)
    pressure = st.slider("Тиск, hPa", 980.0, 1045.0, float(default_pressure), step=0.5)
    wind_speed = st.slider("Швидкість вітру, м/с", 0.0, 25.0, float(default_wind), step=0.5)

    lag_values = None

    if "Прогнозний режим" in mode:
        st.subheader("Поточний стан навантаження")
        st.caption("Ці значення потрібні для моделі з лаговими ознаками.")

        lag_1 = st.number_input("lag_1 (попередня година), MW", min_value=0.0, value=11800.0, step=10.0)
        lag_2 = st.number_input("lag_2, MW", min_value=0.0, value=11750.0, step=10.0)
        lag_3 = st.number_input("lag_3, MW", min_value=0.0, value=11720.0, step=10.0)
        lag_6 = st.number_input("lag_6, MW", min_value=0.0, value=11650.0, step=10.0)
        lag_12 = st.number_input("lag_12, MW", min_value=0.0, value=11400.0, step=10.0)
        lag_24 = st.number_input("lag_24, MW", min_value=0.0, value=11200.0, step=10.0)

        lag_values = prepare_full_lag_values(
            lag_1=lag_1,
            lag_2=lag_2,
            lag_3=lag_3,
            lag_6=lag_6,
            lag_12=lag_12,
            lag_24=lag_24
        )

if "Інтерпретаційний режим" in mode:
    model = no_lags_model
    feature_cols = BASE_FEATURES
else:
    model = full_model
    feature_cols = FULL_FEATURES

current_params = {
    "hour": hour,
    "weekday": weekday,
    "month": month,
    "is_holiday": is_holiday,
    "temp": temp,
    "humidity": humidity,
    "pressure": pressure,
    "wind_speed": wind_speed,
}

current_df = build_input_df(
    hour=hour,
    weekday=weekday,
    month=month,
    is_holiday=is_holiday,
    temp=temp,
    humidity=humidity,
    pressure=pressure,
    wind_speed=wind_speed,
    lag_values=lag_values,
)

baseline_df = build_input_df(
    hour=12,
    weekday=2,
    month=7,
    is_holiday=0,
    temp=20.0,
    humidity=55.0,
    pressure=1015.0,
    wind_speed=3.0,
    lag_values=lag_values if lag_values is not None else None,
)

current_pred = predict_consumption(model, current_df, feature_cols)
baseline_pred = predict_consumption(model, baseline_df, feature_cols)
delta_mw = current_pred - baseline_pred
delta_pct = (delta_mw / baseline_pred * 100) if baseline_pred != 0 else 0.0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Прогнозоване споживання", f"{current_pred:,.2f} MW")

with col2:
    st.metric("Базовий сценарій", f"{baseline_pred:,.2f} MW")

with col3:
    st.metric("Зміна відносно бази", f"{delta_mw:,.2f} MW", f"{delta_pct:.2f}%")

st.subheader("Поточний сценарій")
display_df = pd.DataFrame({
    "Параметр": [
        "Година", "День тижня", "Місяць", "Свято",
        "Температура", "Вологість", "Тиск", "Швидкість вітру"
    ],
    "Значення": [
        hour, weekday, month, is_holiday,
        temp, humidity, pressure, wind_speed
    ]
})
st.dataframe(display_df, use_container_width=True)


left, right = st.columns(2)

with left:
    fig_temp = make_temperature_sensitivity_plot(
        model=model,
        feature_cols=feature_cols,
        current_params=current_params,
        lag_values=lag_values
    )
    st.pyplot(fig_temp)

with right:
    fig_hour = make_hourly_profile_plot(
        model=model,
        feature_cols=feature_cols,
        current_params=current_params,
        lag_values=lag_values
    )
    st.pyplot(fig_hour)


with st.expander("Пояснення режимів"):
    st.markdown("""
**Інтерпретаційний режим** використовує модель без лагових ознак.  
Він краще підходить для демонстрації впливу погодних і часових параметрів, оскільки прогноз сильніше реагує на повзунки.

**Прогнозний режим** використовує повну модель з лаговими та rolling-ознаками.  
Він зазвичай дає точніший прогноз, але потребує поточного контексту навантаження (`lag_1`, `lag_2`, ...).
""")

st.markdown("---")
