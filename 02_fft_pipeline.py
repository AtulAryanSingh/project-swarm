import numpy as np
import matplotlib.pyplot as plt

# 1. Generate the Raw Time Data (1 second of audio/telemetry)
sample_rate = 1000  # 1000 data points per second
time = np.linspace(0, 1, sample_rate)

# The Smart Rock is transmitting at exactly 50 Hz
smart_rock_freq = 50 
clean_signal = np.sin(2 * np.pi * smart_rock_freq * time)

# The Solar Flare hits (Massive random noise)
solar_noise = np.random.normal(0, 2.5, len(time)) 

# The data the Distant Sphere actually receives (It looks like pure static)
raw_telemetry = clean_signal + solar_noise

# 2. THE PIPELINE: Execute the Fast Fourier Transform (FFT)
# This converts the messy Time Array into a structured Frequency Array
fft_output = np.fft.fft(raw_telemetry)

# We need the absolute values (magnitude) of the frequencies
frequencies_magnitude = np.abs(fft_output)

# Create an array of the actual frequency labels (0Hz, 1Hz, 2Hz... up to 500Hz)
freq_labels = np.fft.fftfreq(len(time), 1/sample_rate)

# We only look at the positive frequencies (first half of the array)
half_point = len(freq_labels) // 2
positive_freqs = freq_labels[:half_point]
positive_magnitudes = frequencies_magnitude[:half_point]

# 3. Plotting the results to prove it worked
plt.figure(figsize=(12, 6))
plt.suptitle("Pipeline Output: Time Domain vs Frequency Domain", fontsize=14)

# Top Graph: The Mess
plt.subplot(2, 1, 1)
plt.plot(time, raw_telemetry, color='red', alpha=0.7)
plt.title("What the Sphere Receives (Time Domain - Pure Noise)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Bottom Graph: The Clean Result
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, positive_magnitudes, color='blue')
plt.title("After FFT Routing (Frequency Domain - Isolated Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 100) # Zoom in to see the spike at 50Hz
plt.axvline(x=50, color='green', linestyle='--', label='Smart Rock Frequency (50Hz)')
plt.legend()

plt.tight_layout()
plt.show()
