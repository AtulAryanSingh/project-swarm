import numpy as np
import matplotlib.pyplot as plt

# 1. GENERATE THE COSMIC FIREHOSE
time = np.linspace(0, 1, 500)
beam_1_energy = np.sin(2 * np.pi * 5 * time) * 10      # 5 Hz Heat Wave (Energy)
beam_2_data = np.sin(2 * np.pi * 80 * time) * 5        # 80 Hz Telemetry Data
beam_3_noise = np.random.normal(0, 0.5, 500)           # Chaos / Static

incoming_stream = beam_1_energy + beam_2_data + beam_3_noise

# 2. RUN THE FFT (The Sphere's Brain)
frequencies = np.fft.fftfreq(len(incoming_stream))
amplitudes = np.abs(np.fft.fft(incoming_stream))

positive_freqs = frequencies[:len(frequencies)//2] * 500
positive_amps = amplitudes[:len(amplitudes)//2]

dominant_freq = positive_freqs[np.argmax(positive_amps)]
print(f"Sphere Router: Dominant frequency detected at {dominant_freq:.1f} Hz")

# 3. DRAW THE PIPELINE VISUALIZATION
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Distant Sphere: Stream Ingestion & Frequency Routing", fontsize=16)

ax1.plot(time, incoming_stream, color='crimson', linewidth=1.5)
ax1.set_title("Incoming Firehose (Raw Time-Domain Chaos)", fontsize=12)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)

ax2.plot(positive_freqs, positive_amps, color='cyan', linewidth=2)
ax2.fill_between(positive_freqs, positive_amps, color='cyan', alpha=0.3)
ax2.set_title("FFT Extraction (Finding the Hidden Signals)", fontsize=12)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Signal Strength")
ax2.set_xlim(0, 100)
ax2.grid(True, alpha=0.3)

ax2.axvline(x=5, color='orange', linestyle='--', label='Energy Battery Threshold (5 Hz)')
ax2.axvline(x=80, color='lime', linestyle='--', label='ML Data Threshold (80 Hz)')
ax2.legend()

plt.tight_layout()
plt.show()