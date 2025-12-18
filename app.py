import streamlit as st
import numpy as np
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AdSpend Intelligence",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Styling ----------------
st.markdown("""
<style>
body {
    background-color: #020617;
}
.card {
    background: rgba(30,41,59,0.75);
    backdrop-filter: blur(8px);
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.result {
    background: linear-gradient(135deg,#2563eb,#1e40af);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
}
h1, h2, h3 {
    color: #e5e7eb;
}
p {
    color: #cbd5e1;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("advertising_poly_model.pkl", "rb") as f:
        model, poly = pickle.load(f)
    return model, poly

model, poly = load_model()

# ---------------- Header ----------------
st.markdown("""
<div class="card">
    <h1>AdSpend Intelligence</h1>
    <p>
        Predict expected product sales based on advertising investments
        using <b>Polynomial Regression</b>.
        Designed for strategic planning, not crystal-ball worship.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Input Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("Advertising Budget Allocation")

tv = st.slider(
    "TV Advertising Spend ($)",
    min_value=0.0,
    max_value=300.0,
    value=150.0,
    step=1.0
)

radio = st.slider(
    "Radio Advertising Spend ($)",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    step=1.0
)

newspaper = st.slider(
    "Newspaper Advertising Spend ($)",
    min_value=0.0,
    max_value=120.0,
    value=60.0,
    step=1.0
)

predict = st.button("Predict Sales", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Prediction ----------------
if predict:
    user_input = np.array([[tv, radio, newspaper]])
    user_poly = poly.transform(user_input)
    predicted_sales = model.predict(user_poly)[0]

    st.markdown(
        f"""
        <div class="result">
            <h2>Estimated Sales</h2>
            <p>{predicted_sales:.2f} units</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    Polynomial Regression • Advertising Analytics • Educational Use
</div>
""", unsafe_allow_html=True)
