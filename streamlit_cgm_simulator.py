
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
import time


st.set_page_config(
    page_title="CGM Simulator with AI Predictions",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with your color palette
st.markdown("""
<style>
    :root {
        --primary-color: #7CB342;
        --primary-dark: #689F38;
        --primary-light: #DCEDC8;
        --glucose-red: #E53935;
        --insulin-blue: #039BE5;
        --exercise-orange: #FB8C00;
        --meal-green: #43A047;
        --bg-main: #f8fafc;
        --bg-card: #ffffff;
        --text-main: #333333;
        --text-muted: #666666;
        --border-color: #e0e0e0;
        --success-bg: #E8F5E9;
        --success-text: #2E7D32;
        --error-bg: #FFEBEE;
        --error-text: #C62828;
        --warning-bg: #FFF3E0;
        --warning-text: #EF6C00;
    }
    
    /* Main background */
    .main {
        background-color: var(--bg-main);
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: var(--text-main);
        font-weight: 600;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    div.stAlert[data-baseweb="notification"] > div:first-child {
        border-left-color: var(--error-text);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-main);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border-color);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained hypoglycemia prediction model"""
    try:
        with open('final_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("Model file 'final_model.pkl' not found! Please ensure it's in the same directory.")
        return None

model_package = load_model()

# ── API trick: handle ?api=predict&payload=... before rendering any UI ───────
import json
from urllib.parse import unquote

_params = st.query_params
if _params.get("api") == "predict":
    payload_str = unquote(_params.get("payload", "{}"))
    try:
        raw = json.loads(payload_str).get("features", {})
        row = {col: raw.get(col, 0) for col in feature_columns}
        df = pd.DataFrame([row])
        X_scaled = scaler.transform(df)
        probability = float(model.predict_proba(X_scaled)[0, 1])
        prediction = 1 if probability >= threshold else 0
        risk_level = "LOW" if probability < 0.3 else ("MEDIUM" if probability < 0.7 else "HIGH")
        result = {"prediction": prediction, "probability": round(probability, 4), "risk_level": risk_level}
    except Exception as e:
        result = {"error": str(e)}

    st.write(json.dumps(result))
    st.stop()

if model_package:
    model = model_package['model']
    scaler = model_package['scaler']
    feature_columns = model_package['feature_columns']
    threshold = model_package.get('recommended_threshold', 0.4)

@st.cache_data
def load_patient_data(file_path):
    """Load patient CGM data"""
    df = pd.read_csv(file_path, sep=';')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df

# Load the uploaded patient data
patient_data = load_patient_data('HUPA0028P.csv')

def prepare_features_for_prediction(current_idx, data):
    """Prepare features for ML prediction from historical data"""
    
    # Get recent history (last 60 minutes = 12 readings at 5-min intervals)
    start_idx = max(0, current_idx - 11)
    recent_data = data.iloc[start_idx:current_idx + 1]
    
    features = {}
    
    # Current values
    current = data.iloc[current_idx]
    features['glucose'] = current['glucose']
    features['bolus_volume_delivered'] = current['bolus_volume_delivered']
    features['basal_rate'] = current['basal_rate']
    features['carb_input'] = current['carb_input']
    features['steps'] = current['steps']
    features['calories'] = current['calories']
    features['heart_rate'] = current['heart_rate']
    
    # Temporal features
    hour = current['time'].hour
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['is_weekend'] = 1 if current['time'].dayofweek >= 5 else 0
    features['minutes_since_midnight'] = hour * 60 + current['time'].minute
    
    # Glucose dynamics
    glucose_history = recent_data['glucose'].values
    
    if len(glucose_history) >= 2:
        features['glucose_velocity'] = (glucose_history[-1] - glucose_history[-2]) / 5
    else:
        features['glucose_velocity'] = 0
    
    if len(glucose_history) >= 3:
        velocity_now = (glucose_history[-1] - glucose_history[-2]) / 5
        velocity_prev = (glucose_history[-2] - glucose_history[-3]) / 5
        features['glucose_acceleration'] = velocity_now - velocity_prev
    else:
        features['glucose_acceleration'] = 0
    
    # Rolling statistics
    for window, window_min in [(3, 15), (6, 30), (12, 60)]:
        recent = glucose_history[-window:] if len(glucose_history) >= window else glucose_history
        features[f'glucose_mean_{window_min}min'] = np.mean(recent)
        features[f'glucose_std_{window_min}min'] = np.std(recent) if len(recent) > 1 else 0
        features[f'glucose_min_{window_min}min'] = np.min(recent)
        features[f'glucose_max_{window_min}min'] = np.max(recent)
    
    # Insulin features
    bolus_history = recent_data['bolus_volume_delivered'].values
    basal_history = recent_data['basal_rate'].values
    
    for window, window_min in [(3, 15), (6, 30), (12, 60)]:
        features[f'bolus_sum_{window_min}min'] = np.sum(bolus_history[-window:])
        features[f'basal_sum_{window_min}min'] = np.sum(basal_history[-window:])
    
    features['insulin_on_board'] = np.sum(bolus_history)
    
    # Carb features
    carb_history = recent_data['carb_input'].values
    
    for window, window_min in [(3, 15), (6, 30), (12, 60)]:
        features[f'carb_sum_{window_min}min'] = np.sum(carb_history[-window:])
    
    features['carb_to_insulin_ratio'] = features['carb_sum_60min'] / (features['bolus_sum_60min'] + 0.1)
    
    # Activity features
    steps_history = recent_data['steps'].values
    calories_history = recent_data['calories'].values
    heart_rate_history = recent_data['heart_rate'].values
    
    for window, window_min in [(3, 15), (6, 30), (12, 60)]:
        features[f'steps_sum_{window_min}min'] = np.sum(steps_history[-window:])
        features[f'calories_sum_{window_min}min'] = np.sum(calories_history[-window:])
        features[f'heart_rate_mean_{window_min}min'] = np.mean(heart_rate_history[-window:])
    
    # Create DataFrame
    feature_df = pd.DataFrame([features])
    
    # Ensure all features exist
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    return feature_df[feature_columns]

def make_prediction(features):
    """Make hypoglycemia prediction"""
    if model_package is None:
        return None, None, None
    
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0, 1]
    prediction = 1 if probability >= threshold else 0
    
    # Determine risk level
    if probability < 0.3:
        risk_level = 'LOW'
    elif probability < 0.7:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'HIGH'
    
    return prediction, probability, risk_level

if 'current_index' not in st.session_state:
    st.session_state.current_index = 100  # Start after enough history
    
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
    
if 'manual_entries' not in st.session_state:
    st.session_state.manual_entries = []

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.title("Real-Time CGM Simulator with AI Predictions")
st.markdown("### Patient: HUPA0028P | Continuous Glucose Monitoring with 30-Minute Hypoglycemia Prediction")

with st.sidebar:
    st.header("Simulation Controls")
    
    # Navigation
    st.subheader("Navigate")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start"):
            st.session_state.current_index = 100
            st.rerun()
    
    with col2:
        if st.button("End"):
            st.session_state.current_index = len(patient_data) - 100
            st.rerun()
    
    # Slider to jump to any point
    st.session_state.current_index = st.slider(
        "Time Position",
        min_value=100,
        max_value=len(patient_data) - 50,
        value=st.session_state.current_index,
        step=1
    )
    
    st.markdown("---")
    
    # Real-time simulation
    st.subheader("Real-Time Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Play", key="play"):
            st.session_state.simulation_running = True
            
    with col2:
        if st.button("Pause", key="pause"):
            st.session_state.simulation_running = False
    
    speed = st.select_slider(
        "Simulation Speed",
        options=[0.1, 0.5, 1, 2, 5],
        value=1,
        format_func=lambda x: f"{x}x"
    )
    
    st.markdown("---")
    
    # Manual data entry
    st.subheader("Add Manual Entry")
    
    with st.form("manual_entry"):
        new_glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=120)
        new_insulin = st.number_input("Insulin Bolus (units)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
        new_carbs = st.number_input("Carbs (grams)", min_value=0, max_value=150, value=0)
        
        if st.form_submit_button("Add Entry"):
            current_time = patient_data.iloc[st.session_state.current_index]['time'] + timedelta(minutes=5)
            new_entry = {
                'time': current_time,
                'glucose': new_glucose,
                'bolus_volume_delivered': new_insulin,
                'carb_input': new_carbs,
                'basal_rate': 0.8,
                'steps': 0,
                'calories': 0,
                'heart_rate': 75
            }
            st.session_state.manual_entries.append(new_entry)
            st.success("Entry added successfully")
    
    st.markdown("---")
    
    # Model info
    if model_package:
        st.subheader("Model Information")
        st.write(f"**Model:** {model_package.get('model_name', 'Unknown')}")
        st.write(f"**Threshold:** {threshold:.2f}")
        st.write(f"**Features:** {len(feature_columns)}")
        
        if 'test_performance' in model_package:
            perf = model_package['test_performance']
            st.write(f"**Test Recall:** {perf.get('recall', 0):.1%}")

# Get current data point
current = patient_data.iloc[st.session_state.current_index]

# Make prediction
features = prepare_features_for_prediction(st.session_state.current_index, patient_data)
prediction, probability, risk_level = make_prediction(features)

# Store prediction history
st.session_state.prediction_history.append({
    'time': current['time'],
    'glucose': current['glucose'],
    'probability': probability if probability else 0,
    'risk_level': risk_level if risk_level else 'UNKNOWN'
})

# Keep only last 200 predictions
if len(st.session_state.prediction_history) > 200:
    st.session_state.prediction_history.pop(0)

# Critical alert
if current['glucose'] < 70:
    st.markdown(f"""
    <div style="background-color: var(--error-bg); color: var(--error-text); 
                padding: 1rem; border-radius: 8px; border-left: 4px solid var(--glucose-red);
                margin-bottom: 1rem;">
        <strong>HYPOGLYCEMIA ALERT</strong><br>
        Current glucose: {current['glucose']:.1f} mg/dL - Take 15g fast-acting carbs immediately
    </div>
    """, unsafe_allow_html=True)
elif risk_level == 'HIGH':
    st.markdown(f"""
    <div style="background-color: var(--error-bg); color: var(--error-text); 
                padding: 1rem; border-radius: 8px; border-left: 4px solid var(--glucose-red);
                margin-bottom: 1rem;">
        <strong>HIGH RISK</strong><br>
        Hypoglycemia predicted in 30 minutes (probability: {probability:.1%})
    </div>
    """, unsafe_allow_html=True)
elif risk_level == 'MEDIUM':
    st.markdown(f"""
    <div style="background-color: var(--warning-bg); color: var(--warning-text); 
                padding: 1rem; border-radius: 8px; border-left: 4px solid var(--exercise-orange);
                margin-bottom: 1rem;">
        <strong>MEDIUM RISK</strong><br>
        Monitor closely (probability: {probability:.1%})
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Glucose",
        f"{current['glucose']:.1f} mg/dL",
        delta=f"{current['glucose'] - patient_data.iloc[st.session_state.current_index-1]['glucose']:.1f}" if st.session_state.current_index > 0 else None,
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Risk Level",
        risk_level if risk_level else "N/A",
        delta=f"{probability*100:.1f}%" if probability else "N/A"
    )

with col3:
    # Calculate trend
    if st.session_state.current_index >= 2:
        recent_glucose = patient_data.iloc[st.session_state.current_index-2:st.session_state.current_index+1]['glucose'].values
        trend = recent_glucose[-1] - recent_glucose[0]
        trend_icon = "↑" if trend > 0 else "↓" if trend < 0 else "→"
        st.metric("Trend", trend_icon, delta=f"{trend:.1f} mg/dL")
    else:
        st.metric("Trend", "→", delta="0.0 mg/dL")

with col4:
    st.metric(
        "Recent Insulin",
        f"{current['bolus_volume_delivered']:.1f} u"
    )

with col5:
    st.metric(
        "Recent Carbs",
        f"{current['carb_input']:.0f} g"
    )


st.markdown("---")

# Prepare data for visualization
lookback = 60  # Show last 60 readings (5 hours)
lookahead = 12  # Show next 12 readings (1 hour) as prediction

start_idx = max(0, st.session_state.current_index - lookback)
end_idx = min(len(patient_data), st.session_state.current_index + lookahead)

historical = patient_data.iloc[start_idx:st.session_state.current_index + 1]
future = patient_data.iloc[st.session_state.current_index:end_idx]

# Create figure with subplots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        'Glucose Levels with AI Predictions',
        'Insulin & Carbohydrate Input',
        'Activity (Steps & Heart Rate)'
    ),
    vertical_spacing=0.1,
    row_heights=[0.5, 0.25, 0.25]
)

# ===== PLOT 1: GLUCOSE WITH PREDICTIONS =====

# Historical glucose
fig.add_trace(
    go.Scatter(
        x=historical['time'],
        y=historical['glucose'],
        mode='lines+markers',
        name='Actual Glucose',
        line=dict(color='#E53935', width=3),  # glucose-red
        marker=dict(size=6)
    ),
    row=1, col=1
)

# Current point (highlight)
fig.add_trace(
    go.Scatter(
        x=[current['time']],
        y=[current['glucose']],
        mode='markers',
        name='Current Reading',
        marker=dict(size=15, color='#7CB342', symbol='star'),  # primary-color
        showlegend=True
    ),
    row=1, col=1
)

# Predicted future glucose (simple linear extrapolation + prediction probability)
if len(historical) >= 6:
    # Use velocity to predict future trend
    velocity = (historical['glucose'].iloc[-1] - historical['glucose'].iloc[-6]) / 25  # per minute
    
    future_times = []
    future_glucose_optimistic = []
    future_glucose_pessimistic = []
    future_glucose_likely = []
    
    for i in range(1, 7):  # Predict 30 minutes ahead (6 readings)
        future_time = current['time'] + timedelta(minutes=i*5)
        future_times.append(future_time)
        
        # Likely trajectory
        base_prediction = current['glucose'] + velocity * i * 5
        
        # Adjust based on prediction probability
        if probability and probability > 0.5:
            # High risk - predict drop
            adjustment = -5 * (probability - 0.5) * 2 * i
        else:
            adjustment = 0
        
        future_glucose_likely.append(base_prediction + adjustment)
        future_glucose_optimistic.append(base_prediction + adjustment + 5)
        future_glucose_pessimistic.append(base_prediction + adjustment - 5)
    
    # Predicted range (confidence interval)
    fig.add_trace(
        go.Scatter(
            x=future_times + future_times[::-1],
            y=future_glucose_pessimistic + future_glucose_optimistic[::-1],
            fill='toself',
            fillcolor='rgba(124, 179, 66, 0.2)',  # primary-color with transparency
            line=dict(color='rgba(255,255,255,0)'),
            name='Prediction Range',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Likely prediction
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_glucose_likely,
            mode='lines+markers',
            name='Predicted Glucose',
            line=dict(color='#7CB342', width=2, dash='dash'),  # primary-color
            marker=dict(size=6, symbol='diamond')
        ),
        row=1, col=1
    )

# Threshold lines
fig.add_hline(y=70, line_dash="dash", line_color="#E53935", annotation_text="Hypo Threshold (70)", row=1, col=1)  # glucose-red
fig.add_hline(y=180, line_dash="dash", line_color="#FB8C00", annotation_text="Hyper Threshold (180)", row=1, col=1)  # exercise-orange

# Target range
fig.add_hrect(y0=70, y1=180, fillcolor="#DCEDC8", opacity=0.2, line_width=0, row=1, col=1)  # primary-light


# Insulin
fig.add_trace(
    go.Bar(
        x=historical['time'],
        y=historical['bolus_volume_delivered'],
        name='Insulin Bolus',
        marker_color='#039BE5',  # insulin-blue
        opacity=0.7
    ),
    row=2, col=1
)

# Carbs
fig.add_trace(
    go.Bar(
        x=historical['time'],
        y=historical['carb_input'],
        name='Carbs',
        marker_color='#43A047',  # meal-green
        opacity=0.7
    ),
    row=2, col=1
)


# Steps
fig.add_trace(
    go.Scatter(
        x=historical['time'],
        y=historical['steps'],
        mode='lines',
        name='Steps',
        line=dict(color='#FB8C00', width=2),  # exercise-orange
        fill='tozeroy',
        opacity=0.6
    ),
    row=3, col=1
)

# Heart rate (secondary y-axis effect)
fig.add_trace(
    go.Scatter(
        x=historical['time'],
        y=historical['heart_rate'],
        mode='lines',
        name='Heart Rate',
        line=dict(color='#E53935', width=2),  # glucose-red
        yaxis='y4'
    ),
    row=3, col=1
)

# Update layout
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1, range=[40, 250])
fig.update_yaxes(title_text="Units / Grams", row=2, col=1)
fig.update_yaxes(title_text="Steps", row=3, col=1)

fig.update_layout(
    height=1000,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.subheader("Hypoglycemia Risk Probability Over Time")

if len(st.session_state.prediction_history) > 1:
    pred_df = pd.DataFrame(st.session_state.prediction_history)
    
    fig_prob = go.Figure()
    
    # Probability line
    fig_prob.add_trace(
        go.Scatter(
            x=pred_df['time'],
            y=pred_df['probability'] * 100,
            mode='lines+markers',
            name='Risk Probability',
            line=dict(color='#E53935', width=3),  # glucose-red
            fill='tozeroy',
            fillcolor='rgba(229, 57, 53, 0.2)'
        )
    )
    
    # Threshold lines
    fig_prob.add_hline(y=30, line_dash="dash", line_color="#7CB342", annotation_text="Low Risk (30%)")  # primary-color
    fig_prob.add_hline(y=70, line_dash="dash", line_color="#E53935", annotation_text="High Risk (70%)")  # glucose-red
    
    # Risk zones
    fig_prob.add_hrect(y0=0, y1=30, fillcolor="#E8F5E9", opacity=0.3, line_width=0)  # success-bg
    fig_prob.add_hrect(y0=30, y1=70, fillcolor="#FFF3E0", opacity=0.3, line_width=0)  # warning-bg
    fig_prob.add_hrect(y0=70, y1=100, fillcolor="#FFEBEE", opacity=0.3, line_width=0)  # error-bg
    
    fig_prob.update_layout(
        xaxis_title="Time",
        yaxis_title="Probability (%)",
        height=300,
        template='plotly_white',
        hovermode='x'
    )
    
    st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Session Statistics")
    
    session_data = patient_data.iloc[100:st.session_state.current_index + 1]
    
    st.write(f"**Readings:** {len(session_data)}")
    st.write(f"**Duration:** {(session_data['time'].max() - session_data['time'].min()).total_seconds() / 3600:.1f} hours")
    st.write(f"**Average Glucose:** {session_data['glucose'].mean():.1f} mg/dL")
    st.write(f"**Time in Range (70-180):** {((session_data['glucose'] >= 70) & (session_data['glucose'] <= 180)).mean() * 100:.1f}%")
    st.write(f"**Time Below 70:** {(session_data['glucose'] < 70).mean() * 100:.1f}%")
    st.write(f"**Time Above 180:** {(session_data['glucose'] > 180).mean() * 100:.1f}%")

with col2:
    st.subheader("Prediction Accuracy")
    
    if len(st.session_state.prediction_history) > 6:
        # Calculate how many predictions were correct
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(st.session_state.prediction_history) - 6):
            pred = st.session_state.prediction_history[i]
            # Check actual glucose 6 readings later (30 minutes)
            if i + 6 < len(st.session_state.prediction_history):
                actual_future = st.session_state.prediction_history[i + 6]
                predicted_hypo = pred['probability'] >= threshold
                actual_hypo = actual_future['glucose'] < 70
                
                if predicted_hypo == actual_hypo:
                    correct_predictions += 1
                total_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions * 100
            st.write(f"**Predictions Made:** {total_predictions}")
            st.write(f"**Correct Predictions:** {correct_predictions}")
            st.write(f"**Accuracy:** {accuracy:.1f}%")
        else:
            st.write("Not enough data yet...")
    else:
        st.write("Not enough predictions yet...")


st.markdown("---")
st.subheader("Recent Readings")

# Show last 10 readings
recent_readings = patient_data.iloc[max(0, st.session_state.current_index - 9):st.session_state.current_index + 1].copy()
recent_readings['time'] = recent_readings['time'].dt.strftime('%Y-%m-%d %H:%M')

st.dataframe(
    recent_readings[['time', 'glucose', 'bolus_volume_delivered', 'carb_input', 'steps', 'heart_rate']],
    use_container_width=True,
    hide_index=True
)

if st.session_state.simulation_running:
    time.sleep(1 / speed)  # Adjust speed
    
    if st.session_state.current_index < len(patient_data) - 50:
        st.session_state.current_index += 1
        st.rerun()
    else:
        st.session_state.simulation_running = False
        st.success("Simulation complete")

st.markdown("---")
st.caption("AI-Powered Hypoglycemia Prediction System | Predicts 30 minutes ahead | Real-time CGM Simulation")
