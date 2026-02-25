import numpy as np
import matplotlib.pyplot as plt
import hashlib

def prompt_to_signal(t, prompt, freq_base=1.0):
    """
    Convertit le texte du prompt en un signal temporel.
    Utilise un hash MD5 pour moduler amplitude et phase.
    """
    # Empreinte numérique du prompt (entre 0 et 1)
    hash_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16) / 2**128
    # Amplitude modulée par le hash, fréquence de base
    return 0.5 * np.sin(2 * np.pi * freq_base * t + 2 * np.pi * hash_val)

def query_signal(t, freq=0.8):
    """Signal de test (pour compatibilité)"""
    return 1.2 * np.sin(freq * t) + 0.5 * np.cos(2.1 * t)

def balanced_signal(t):
    """Signal de test avec bruit (pour compatibilité)"""
    freq = 0.8 if (int(t/10) % 2 == 0) else 0.4
    return 1.5 * np.sin(freq * t) + 0.2 * np.random.normal()

def plot_attractor(pc, pd, title="Attracteur TTU-MC3", color='#00ff41'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pc, pd, color=color, lw=0.5, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Cohérence (Φc)")
    ax.set_ylabel("Dissipation (Φd)")
    ax.grid(alpha=0.2)
    return fig

def plot_time_series(t, pm, pc, pd):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, pm, color='cyan', lw=0.8)
    axes[0].set_ylabel("Mémoire (Φm)")
    axes[0].grid(alpha=0.2)
    axes[1].plot(t, pc, color='magenta', lw=0.8)
    axes[1].set_ylabel("Cohérence (Φc)")
    axes[1].grid(alpha=0.2)
    axes[2].plot(t, pd, color='yellow', lw=0.8)
    axes[2].set_ylabel("Dissipation (Φd)")
    axes[2].set_xlabel("Temps")
    axes[2].grid(alpha=0.2)
    plt.tight_layout()
    return fig