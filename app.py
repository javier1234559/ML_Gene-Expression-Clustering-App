import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

# ============================================================================
# DATA LAYER
# ============================================================================

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    x = np.linspace(0, 100, 80)
    y = 2.5 * x + 10 + np.random.normal(0, 12, 80)
    return pd.DataFrame({'X': x, 'Y': y})

# ============================================================================
# ML LAYER
# ============================================================================

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = {
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'mae': np.abs(y - y_pred).mean()
    }
    
    return model, y_pred, metrics

def predict_value(model, x_value):
    return model.predict([[x_value]])[0]

# ============================================================================
# VISUALIZATION LAYER
# ============================================================================

def create_scatter_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.iloc[:, 0], 
        y=df.iloc[:, 1],
        mode='markers',
        marker=dict(size=8, color='#95a5a6', opacity=0.6)
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        template='plotly_white',
        xaxis_title=df.columns[0],
        yaxis_title=df.columns[1]
    )
    return fig

def create_regression_plot(X, y, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=y,
        mode='markers',
        name='Actual',
        marker=dict(size=8, color='#3498db', opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=y_pred,
        mode='lines',
        name='Prediction',
        line=dict(color='#e74c3c', width=3)
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        template='plotly_white',
        showlegend=True
    )
    return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    st.title("🤖 Machine Learning Demo")
    st.caption("Linear Regression - Clean & Simple")
    st.divider()

def render_data_source():
    st.subheader("📁 Data Source")
    
    tab1, tab2 = st.tabs(["📤 Upload", "📊 Sample"])
    
    with tab1:
        file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
        if file:
            st.session_state.data = pd.read_csv(file)
            st.success("✅ Loaded")
    
    with tab2:
        if st.button("Load Sample Data", use_container_width=True):
            st.session_state.data = load_sample_data()
            st.success("✅ Loaded")

def render_data_preview(df):
    st.divider()
    st.subheader("👁️ Data Preview")
    
    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Cols", len(df.columns))
    
    st.dataframe(df.head(10), use_container_width=True, height=250)

def render_visualization(df):
    st.subheader("� Visualization")
    
    if st.session_state.get('trained'):
        fig = create_regression_plot(
            st.session_state.X,
            st.session_state.y_data,
            st.session_state.y_pred
        )
    else:
        fig = create_scatter_plot(df)
    
    st.plotly_chart(fig, use_container_width=True)

def render_model_config(df):
    st.subheader("⚙️ Model Config")
    
    x_col = st.selectbox("X (input)", df.columns, key="select_x")
    y_col = st.selectbox("Y (output)", df.columns, index=1, key="select_y")
    
    if st.button("🚀 Train Model", type="primary"):
        X = df[[x_col]].values
        y = df[y_col].values
        
        model, y_pred, metrics = train_model(X, y)
        
        st.session_state.model = model
        st.session_state.X = X
        st.session_state.y_data = y
        st.session_state.y_pred = y_pred
        st.session_state.metrics = metrics
        st.session_state.trained = True
        st.session_state.x_col = x_col
        st.session_state.y_col = y_col
        
        st.session_state.history.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'r2': metrics['r2'],
            'mse': metrics['mse']
        })
        st.rerun()

def render_metrics():
    st.subheader("📊 Metrics")
    
    if st.session_state.get('trained'):
        metrics = st.session_state.metrics
        model = st.session_state.model
        
        st.metric("R² Score", f"{metrics['r2']:.3f}")
        st.metric("MSE", f"{metrics['mse']:.1f}")
        st.metric("MAE", f"{metrics['mae']:.1f}")
        
        slope = model.coef_[0]
        intercept = model.intercept_
        st.caption(f"y = {slope:.2f}x + {intercept:.2f}")
    else:
        st.info("Train model first")

def render_prediction():
    st.subheader("🎯 Prediction")
    
    if st.session_state.get('trained'):
        model = st.session_state.model
        X = st.session_state.X
        x_col = st.session_state.x_col
        y_col = st.session_state.y_col
        
        new_x = st.number_input(
            f"Enter {x_col} value:",
            value=float(X.mean()),
            step=1.0
        )
        
        prediction = predict_value(model, new_x)
        st.success(f"**Predicted {y_col}:** {prediction:.2f}")
        
        if st.button("📥 Download Results"):
            df = st.session_state.data
            results = df.copy()
            results['Predicted'] = st.session_state.y_pred
            csv = results.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "results.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.info("Train model first")

def render_history():
    if st.session_state.history:
        st.divider()
        st.subheader("📜 Training History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="ML Demo", page_icon="🤖", layout="wide")
    
    st.markdown("""
    <style>
        .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        [data-testid="stMetricValue"] {font-size: 20px;}
        .stButton>button {width: 100%; border-radius: 5px; height: 45px;}
    </style>
    """, unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    render_header()
    
    left_col, right_col = st.columns([1, 1.5])
    
    with left_col:
        render_data_source()
        if st.session_state.data is not None:
            render_data_preview(st.session_state.data)
    
    with right_col:
        if st.session_state.data is not None:
            render_visualization(st.session_state.data)
        else:
            st.subheader("📈 Visualization")
            st.info("👈 Load data to see visualization")
    
    st.divider()
    
    if st.session_state.data is not None:
        col1, col2, col3 = st.columns([1, 1, 1.5])
        
        with col1:
            render_model_config(st.session_state.data)
        
        with col2:
            render_metrics()
        
        with col3:
            render_prediction()
    
    render_history()

if __name__ == "__main__":
    main()
