import streamlit as st
import numpy as np

from spruce_budworm_model import (
    spruce_budworm,
    plot_spruce_budworm_rate,
    evolve_spruce_budworm,
    plot_spruce_budworm,
)

st.title("Spruce Budworm Population Dynamics")

# Sidebar sliders
r = st.sidebar.slider("Growth rate r", 0.0, 1.0, 0.5)
k = st.sidebar.slider("Carrying capacity k", 0.1, 10.0, 10.0)
t_eval = st.sidebar.slider("Time to evolve", 1, 100, 10)

# Initial population automatically k/10
x0 = k / 10.0

# Display the equation with current parameters
st.latex(
    rf"\frac{{dx}}{{dt}} = {r:.3f}x\left(1-\frac{{x}}{{{k:.3f}}}\right) - \frac{{x^2}}{{1+x^2}}"
)

# Session state init (so it doesn't reset)
if "t" not in st.session_state or "x" not in st.session_state:
    st.session_state["t"] = np.array([0.0])
    st.session_state["x"] = np.array([x0])

# Reset button (helpful)
if st.sidebar.button("Reset simulation"):
    st.session_state["t"] = np.array([0.0])
    st.session_state["x"] = np.array([x0])

# Evolve button
if st.sidebar.button("Evolve Forward"):
    t_new, x_new = evolve_spruce_budworm(
    st.session_state["t"],
    st.session_state["x"],
    r=r,
    k=k,
    t_eval=t_eval
)

    st.session_state["t"] = t_new
    st.session_state["x"] = x_new

t = st.session_state["t"]
x = st.session_state["x"]

# Plots
fig1, ax1 = plot_spruce_budworm_rate(x_t=x[-1], r=r, k=k)
st.pyplot(fig1)

fig2, ax2 = plot_spruce_budworm(t, x)
st.pyplot(fig2)

st.caption(f"Current x = {x[-1]:.4f}, time = {t[-1]:.2f}")
