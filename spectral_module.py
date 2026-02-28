# =====================================================
# B2 — SPECTRAL_ANALYSIS_MODULE
# Analyse spectrale cognitive (STFT)
# Compatible : V11 / V13 / S+
# =====================================================

import numpy as np
import streamlit as st

# -----------------------------------------------------
# Import optionnel (sécurisé)
# -----------------------------------------------------

try:
    from scipy.signal import stft
    import matplotlib.pyplot as plt
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False


# -----------------------------------------------------
# SIGNAL DEPUIS LA TIMELINE
# -----------------------------------------------------

def build_signal_from_timeline(word, cortex):
    """
    Construit un signal binaire :
    1 si le mot apparaît dans la timeline
    0 sinon
    """

    timeline = cortex.get("timeline", [])

    if not timeline:
        return np.array([])

    signal = np.array([1 if w == word else 0 for w in timeline])

    return signal


# -----------------------------------------------------
# ANALYSE STFT
# -----------------------------------------------------

def spectral_analysis(word, cortex, nperseg=128):

    if not SPECTRAL_AVAILABLE:
        return {"error": "scipy ou matplotlib non installés"}

    signal = build_signal_from_timeline(word, cortex)

    if len(signal) < nperseg:
        return {
            "error": f"Signal trop court ({len(signal)} < {nperseg})"
        }

    fs = 1.0  # 1 échantillon = 1 mot

    f, t, Zxx = stft(
        signal,
        fs,
        window="blackmanharris",
        nperseg=nperseg,
        noverlap=nperseg // 2
    )

    # -------------------------------------------------
    # Spectrogramme
    # -------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(10,4))

    ax1.pcolormesh(
        t,
        f,
        20*np.log10(np.abs(Zxx)+1e-10),
        shading="gouraud"
    )

    ax1.set_ylabel("Fréquence [cycles/mot]")
    ax1.set_xlabel("Temps [mot]")
    ax1.set_title(f"Spectrogramme — {word}")

    # -------------------------------------------------
    # Fréquence dominante
    # -------------------------------------------------

    mean_amp = np.mean(np.abs(Zxx), axis=1)

    idx_max = np.argmax(mean_amp[1:]) + 1

    freq_dom = f[idx_max]

    phase = np.unwrap(np.angle(Zxx[idx_max, :]))

    omega = 2 * np.pi * freq_dom

    # -------------------------------------------------
    # Amortissement alpha
    # -------------------------------------------------

    peak = mean_amp[idx_max]
    half = peak / np.sqrt(2)

    left = np.where(mean_amp[:idx_max] <= half)[0]
    right = np.where(mean_amp[idx_max:] <= half)[0]

    if len(left) and len(right):
        bandwidth = f[idx_max + right[0]] - f[left[-1]]
        alpha = bandwidth / 2
    else:
        alpha = 0.0

    # -------------------------------------------------
    # Figure phase
    # -------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(t, phase)
    ax2.set_title("Phase déroulée")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel("Phase [rad]")

    # -------------------------------------------------
    # Linéarité phase
    # -------------------------------------------------

    if len(t) > 1:
        coeffs = np.polyfit(t, phase, 1)
        trend = np.polyval(coeffs, t)
        residuals = phase - trend
        linearity = 1 - np.std(residuals)/(np.std(phase)+1e-10)
    else:
        linearity = 0

    results = {
        "freq_dominant": float(freq_dom),
        "omega": float(omega),
        "alpha": float(alpha),
        "lambda": complex(-alpha, omega),
        "linearity": float(linearity),
        "signal_length": int(len(signal)),
        "nperseg": int(nperseg)
    }

    return {
        "results": results,
        "figures": (fig1, fig2)
    }


# -----------------------------------------------------
# UI STREAMLIT (OPTIONNEL)
# -----------------------------------------------------

def spectral_ui(cortex, fragments):

    st.subheader("🔬 Analyse Spectrale Cognitive")

    if not SPECTRAL_AVAILABLE:
        st.warning(
            "Installer : pip install scipy matplotlib"
        )
        return

    if not fragments:
        st.info("Aucun mot disponible.")
        return

    word = st.selectbox("Mot à analyser", fragments)

    nperseg = st.slider(
        "Fenêtre STFT",
        32,
        512,
        128,
        step=32
    )

    if st.button("Lancer analyse spectrale"):

        with st.spinner("Analyse..."):

            output = spectral_analysis(word, cortex, nperseg)

            if "error" in output:
                st.error(output["error"])
                return

            res = output["results"]
            fig1, fig2 = output["figures"]

            c1,c2,c3,c4 = st.columns(4)

            c1.metric("Fréquence", f"{res['freq_dominant']:.4f}")
            c2.metric("Omega", f"{res['omega']:.4f}")
            c3.metric("Alpha", f"{res['alpha']:.4f}")
            c4.metric("Linéarité", f"{res['linearity']:.2f}")

            st.pyplot(fig1)
            st.pyplot(fig2)