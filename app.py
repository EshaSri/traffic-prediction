import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(page_title="Traffic Congestion Prediction System", layout="wide")


# -----------------------------
# 1. Generate dataset
# -----------------------------
@st.cache_data
def create_data():
    data = []

    for _ in range(700):
        hour = random.randint(0, 23)
        weather = random.choice(["Sunny", "Cloudy", "Rainy", "Stormy"])
        road_density = random.randint(10, 100)

        congestion_score = 0

        # Hour effect
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            congestion_score += random.randint(30, 45)
        elif 11 <= hour <= 16:
            congestion_score += random.randint(15, 25)
        else:
            congestion_score += random.randint(5, 12)

        # Weather effect
        if weather == "Sunny":
            congestion_score += random.randint(0, 5)
        elif weather == "Cloudy":
            congestion_score += random.randint(5, 10)
        elif weather == "Rainy":
            congestion_score += random.randint(15, 25)
        elif weather == "Stormy":
            congestion_score += random.randint(25, 35)

        # Road density effect
        congestion_score += int(road_density * 0.45)

        congestion_score = min(congestion_score, 100)

        data.append([hour, weather, road_density, congestion_score])

    df = pd.DataFrame(
        data,
        columns=["Hour", "Weather", "Road_Density", "Congestion_Score"]
    )
    return df


df = create_data()


# -----------------------------
# 2. Preprocessing + ANN model
# -----------------------------
@st.cache_resource
def train_model(dataframe):
    df_model = dataframe.copy()

    weather_encoder = LabelEncoder()
    df_model["Weather_Encoded"] = weather_encoder.fit_transform(df_model["Weather"])

    X = df_model[["Hour", "Weather_Encoded", "Road_Density"]]
    y = df_model["Congestion_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=2500,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 100)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, scaler, weather_encoder, X_test, y_test, y_pred, mae, r2


model, scaler, weather_encoder, X_test, y_test, y_pred, mae, r2 = train_model(df)


# -----------------------------
# 3. Fuzzy logic
# -----------------------------
def low_membership(x):
    if x <= 30:
        return 1
    elif 30 < x < 50:
        return (50 - x) / 20
    return 0


def medium_membership(x):
    if 30 < x < 50:
        return (x - 30) / 20
    elif 50 <= x <= 70:
        return 1
    elif 70 < x < 85:
        return (85 - x) / 15
    return 0


def high_membership(x):
    if 70 < x < 85:
        return (x - 70) / 15
    elif x >= 85:
        return 1
    return 0


def fuzzy_classification(score):
    low = low_membership(score)
    medium = medium_membership(score)
    high = high_membership(score)

    memberships = {
        "LOW": low,
        "MEDIUM": medium,
        "HIGH": high
    }

    label = max(memberships, key=memberships.get)
    return label, memberships


# -----------------------------
# 4. UI
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #1f2937;'>
    Traffic Congestion Prediction System
    </h1>
    <p style='text-align: center; color: #4b5563; font-size:18px;'>
    ANN + Fuzzy Logic Based Intelligent Traffic Prediction
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Traffic Inputs")

    hour = st.slider("Hour of the Day", 0, 23, 8)
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Stormy"])
    road_density = st.slider("Road Density", 0, 100, 50)

    predict_btn = st.button("Predict Traffic")

with col2:
    st.subheader("Model Information")
    st.info("Model Used: Artificial Neural Network (MLP Regressor)")
    st.info("Post-processing: Fuzzy Logic Classification")
    st.info(f"Model MAE: {mae:.2f}")
    st.info(f"Model R² Score: {r2:.2f}")


# -----------------------------
# 5. Prediction
# -----------------------------
if predict_btn:
    weather_encoded = weather_encoder.transform([weather])[0]

    input_df = pd.DataFrame(
        [[hour, weather_encoded, road_density]],
        columns=["Hour", "Weather_Encoded", "Road_Density"]
    )

    input_scaled = scaler.transform(input_df)
    predicted_score = model.predict(input_scaled)[0]
    predicted_score = float(np.clip(predicted_score, 0, 100))

    label, memberships = fuzzy_classification(predicted_score)

    st.markdown("## Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success(f"Predicted Congestion Score: {predicted_score:.2f}")

    with c2:
        if label == "LOW":
            st.success(f"Traffic Level: {label}")
        elif label == "MEDIUM":
            st.warning(f"Traffic Level: {label}")
        else:
            st.error(f"Traffic Level: {label}")

    with c3:
        st.info(
            f"LOW: {memberships['LOW']:.2f} | "
            f"MEDIUM: {memberships['MEDIUM']:.2f} | "
            f"HIGH: {memberships['HIGH']:.2f}"
        )

    # Membership bar chart
    st.subheader("Fuzzy Membership Graph")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    categories = ["LOW", "MEDIUM", "HIGH"]
    values = [memberships["LOW"], memberships["MEDIUM"], memberships["HIGH"]]
    ax1.bar(categories, values)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Membership Value")
    ax1.set_title("Fuzzy Membership Levels")
    st.pyplot(fig1)

    # Traffic interpretation
    st.subheader("Interpretation")
    if label == "LOW":
        st.write("Traffic is expected to be smooth with low congestion.")
    elif label == "MEDIUM":
        st.write("Traffic is moderate. Some congestion may occur.")
    else:
        st.write("Traffic is highly congested. Delay is likely.")

# -----------------------------
# 6. Dataset preview
# -----------------------------
st.markdown("---")
st.subheader("Sample Traffic Dataset")
st.dataframe(df.head(20), use_container_width=True)


# -----------------------------
# 7. Visualizations
# -----------------------------
st.markdown("---")
st.subheader("Traffic Analysis Graphs")

graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    st.write("Average Congestion Score by Weather")
    avg_weather = df.groupby("Weather")["Congestion_Score"].mean().reindex(
        ["Sunny", "Cloudy", "Rainy", "Stormy"]
    )

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(avg_weather.index, avg_weather.values)
    ax2.set_ylabel("Average Congestion Score")
    ax2.set_title("Weather vs Congestion")
    st.pyplot(fig2)

with graph_col2:
    st.write("Road Density vs Congestion Score")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(df["Road_Density"], df["Congestion_Score"], alpha=0.5)
    ax3.set_xlabel("Road Density")
    ax3.set_ylabel("Congestion Score")
    ax3.set_title("Road Density vs Congestion")
    st.pyplot(fig3)


# -----------------------------
# 8. Hour-based visualization
# -----------------------------
st.subheader("Average Congestion by Hour")
avg_hour = df.groupby("Hour")["Congestion_Score"].mean()

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(avg_hour.index, avg_hour.values, marker="o")
ax4.set_xlabel("Hour")
ax4.set_ylabel("Average Congestion Score")
ax4.set_title("Hour vs Average Congestion")
st.pyplot(fig4)


# -----------------------------
# 9. Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit, Artificial Neural Network, and Fuzzy Logic.")
