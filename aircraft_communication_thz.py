# -*- coding: utf-8 -*-

### Air to Ground Aircraft Communication Modelling and Analysis for THz frequency range


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def H_Spr(f, r):
  c = 3e8  # Speed of light
  return c / (4 * np.pi * f * r)

def H_LoS(f, r, t_los):
  return H_Spr(f, r) * np.exp(-1j * 2 * np.pi * f * t_los)

# Example parameters
r = 500  # Distance between transmitter and receiver (meters)
t_los = r / 3e8  # Time delay for line-of-sight propagation

# Define frequency range (0.6 THz to 1 THz)
frequencies = np.linspace(6e11, 1e12, 100)  # Frequency in Hz

# Compute H_LoS(f) for each frequency
H_LoS_values = [H_LoS(f, r, t_los) for f in frequencies]

# Compute magnitude and phase of H_LoS
H_LoS_magnitude = np.abs(H_LoS_values)
H_LoS_phase = np.angle(H_LoS_values)

# Plot magnitude
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(frequencies / 1e9, H_LoS_magnitude)
plt.xlabel('Frequency (GHz)')
plt.ylabel('|H_LoS(f)|')
plt.title('Magnitude of H_LoS vs Frequency')
plt.grid(True)

# Plot phase
plt.subplot(2, 1, 2)
plt.plot(frequencies / 1e9, H_LoS_phase)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Phase of H_LoS(f) (radians)')
plt.title('Phase of H_LoS vs Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

def H_Ref(f, c, r1, r2, r, tau_los, R_f):
    """
    Calculates H_Ref(f) according to the given formula, with tau_ref calculation.
    """
    tau_ref = tau_los + (r1 + r2 - r) / c
    term1 = c / (4 * np.pi * f * (r1 + r2))
    term2 = np.exp(-1j * 2 * np.pi * f * tau_ref - 0.5 * (r1 + r2))
    term3 = R_f(f)
    H_ref = term1 * term2 * term3
    return H_ref

def n_t(f, epsilon_inf=4, epsilon_s=10, tau=1e-12, u_r = 1):
    """
    Computes the frequency-dependent refractive index using a simple Debye model.
    :param f: Frequency (Hz)
    :param epsilon_inf: High-frequency permittivity
    :param epsilon_s: Static permittivity
    :param tau: Relaxation time (s)
    :return: Refractive index n_t(f)
    """

    omega = 2 * np.pi * f
    epsilon_r = epsilon_inf + (epsilon_s - epsilon_inf) / (1 + 1j * omega * tau)
    ref_t = np.sqrt(epsilon_r*u_r)*3e8
    return np.real(ref_t)

def gamma_te(f, r1, r2, r, n_t):
    cos_theta_i = (r1**2 + r2**2 - r**2) / (2 * r1 * r2)
    # theta_i = 0.5 * np.arccos(cos_theta_i)
    res = -np.exp(-2 * cos_theta_i / np.sqrt(n_t**2 - 1))
    return res

def rho(f, sigma, r1, r2, r, c):
    cos_theta_i = (r1**2 + r2**2 - r**2) / (2 * r1 * r2)
    # theta_i = 0.5 * np.arccos(cos_theta_i)
    res = np.exp((-8 * np.pi**2 * f**2 * sigma**2 *cos_theta_i**2) / (c**2))
    return res

def R_f(f, r1, r2, r, n_t, sigma, c):
    return gamma_te(f, r1, r2, r, n_t) * rho(f, sigma, r1, r2, r, c)

# Example usage and test cases:
c = 3e8  # Speed of light
r1 = 100
r2 = 150
r = 200
tau_los = c/r

sigma = 1e-3
relative_permittivity = 4.0
conductivity = 0.01

frequencies = np.linspace(6e11, 1e12, 100)  # Range of frequencies

H_ref_values = []
gamma_te_values = []
rho_values = []
R_f_values = []

for f in frequencies:
    ref_t = n_t(f, epsilon_inf=4, epsilon_s=10, tau=1e-12, u_r = 1)
    H_ref_values.append(H_Ref(f, c, r1, r2, r, tau_los, lambda f: R_f(f, r1, r2, r, ref_t, sigma, c)))
    gamma_te_values.append(gamma_te(f, r1, r2, r, ref_t))

H_ref_magnitudes = np.abs(H_ref_values)
H_ref_phases = np.angle(H_ref_values)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(frequencies, H_ref_magnitudes)
plt.title("Magnitude of H_ref(f)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H_ref(f)|")

plt.subplot(2, 2, 2)
plt.plot(frequencies, H_ref_phases)
plt.title("Phase of H_ref(f)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")

# Convert gamma_te to dB for frequency plot
gamma_te_dB = 20 * np.log10(np.abs(gamma_te_values))

# Plot gamma_te in dB vs frequency
plt.figure(figsize=(8, 6))
plt.plot(frequencies, gamma_te_dB, color='r')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gamma_TE (dB)")
plt.title("Gamma_TE(f) in dB vs Frequency")
plt.grid(True)
plt.show()

def H_Sca(f, rho_0, L, F, lx, ly, g, theta_1):
    """
    Computes the scattering function S(f) based on the modified Beckmann-Kirchhoff theory.
    :param f: Frequency (Hz)
    :param rho_0: Roughness parameter
    :param L: Characteristic length scale (m)
    :param F: Fresnel coefficient
    :param lx: Correlation length in x-direction (m)
    :param ly: Correlation length in y-direction (m)
    :param g: RMS surface slope
    :param theta_1: Incident zenith angle (radians)
    :return: Computed scattering function S(f)
    """

    m_values = np.arange(1, 50)  # Truncate infinite sum at m=50
    v_s = 1  # Example scattering coefficient, modify as needed

    factor_1 = -np.exp(- (2 * np.cos(theta_1)) / (np.sqrt(n_t(f, epsilon_inf=4, epsilon_s=10, tau=1e-12, u_r = 1)**2 - 1)))
    factor_2 = (1 / np.sqrt(1 + g + (g**2/2) + (g**3 / 6)))
    factor_3 = np.sqrt(rho_0**2 + (np.pi * np.cos(theta_1) / 100)*(g*np.exp(-g*v_s) + g**2*np.exp(-v_s/2)/4))

    S_f = factor_1 * factor_2 * factor_3
    return S_f

# Define frequency range
frequencies = np.linspace(6e11, 1e12, 100)  # From 600 GHz to 1 THz

# Compute scattering function S(f) and refractive index n_t(f)
S_values = np.array([H_Sca(f, rho_0=0.1, L=0.05, F=0.9, lx=0.1, ly=0.1, g=0.5, theta_1=np.pi/4) for f in frequencies]) # Check constants
n_values = np.array([n_t(f, epsilon_inf=2, epsilon_s=2, tau=1e-12, u_r=1) for f in frequencies]) # Change epsilon_s

# Convert scattering function to dB scale: S_dB = 20 * log10(|S(f)|)
S_values_dB = 20 * np.log10(np.abs(S_values))

# Plot Scattering Function in dB
plt.figure(figsize=(8, 5))
plt.plot(frequencies / 1e9, S_values_dB, label="Scattering Function (dB)", color='r')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Scattering Function S(f) [dB]")
plt.title("Scattering Function vs Frequency (dB Scale)")
plt.grid()
plt.legend()
plt.show()

# Plot Refractive Index vs Frequency
plt.figure(figsize=(8, 5))
plt.plot(frequencies / 1e9, np.real(n_values), label="Re(n)", color='b')  # Real part of refractive index
plt.plot(frequencies / 1e9, np.imag(n_values), label="Im(n)", color='g', linestyle='dashed')  # Imaginary part
plt.xlabel("Frequency (GHz)")
plt.ylabel("Refractive Index n(f)")
plt.title("Refractive Index vs Frequency")
plt.grid()
plt.legend()
plt.show()

def delta_d(h_d, d1, d2):
    """Calculate additional distance Δd."""
    return (h_d**2 * (d1 + d2)) / (2 * d1 * d2)

def diffraction_angle(h_d, d1, d2):
    """Calculate diffraction angle θ_d in degrees."""
    return 180 - np.degrees(np.arccos(h_d / d1) - np.arccos(h_d / d2))

def v_f(f, h_d, d1, d2, c=3e8):
    """Compute v(f)."""
    delta_d_value = delta_d(h_d, d1, d2)
    return np.sqrt(2 * f * delta_d_value / c)

def L_f(v_f):
  u_1 = 1
  u_2 = 1
  u_3 = 1

  """Compute the diffraction coefficient L(f) based on v(f)."""
  if v_f < 1:
      return u_1 * 0.5 * np.exp(-0.95 * v_f)
  elif 1 <= v_f <= 2.4:
      return u_2 * (0.4 - np.sqrt(0.12 - (0.38 - 0.1 * v_f)**2))
  else:
      return u_3 * 0.225 / v_f

def H_Dif(f, d1, d2, h_d, c=3e8):
    """Compute the diffraction transfer function H_Dif(f)."""
    delta_d_value = delta_d(h_d, d1, d2)
    v = v_f(f, h_d, d1, d2, c)
    L = L_f(v)
    exponent = -1j * (2 * np.pi * f * (d1 + d2) / c + 0.5 * (2 * np.pi * f / c) * delta_d_value)
    return (c / (4 * np.pi * f * (d1 + d2))) * np.exp(exponent) * L

# Example parameters
h_d = 10  # Height of the diffraction point (in meters)
d1 = 4    # Distance from the source to the diffraction point (in meters)
d2 = 5    # Distance from the diffraction point to the receiver (in meters)

# Frequency range (from 1 GHz to 10 GHz)
frequencies = np.linspace(1e9, 10e9, 1000)  # Frequency range (Hz)

# Compute H_Dif(f) for each frequency
H_Dif_values = [H_Dif(f, d1, d2, h_d) for f in frequencies]

# Plot the magnitude of H_Dif(f)
H_Dif_magnitude = np.abs(H_Dif_values)

plt.figure(figsize=(8, 6))
plt.plot(frequencies / 1e9, H_Dif_magnitude)  # Convert frequencies to GHz for plotting
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude of H_Dif(f)')
plt.title('Diffraction Transfer Function H_Dif(f) vs Frequency')
plt.grid(True)
plt.show()

def coherence_bandwidth(h_i_values, tau_values):
    tau_mean = np.sum(h_i_values*tau_values)/np.sum(h_i_values)
    tau_mean_square = np.sum(h_i_values*(tau_values**2))/np.sum(h_i_values)
    delay_spread = np.sqrt(abs(tau_mean_square - tau_mean**2))
    coherence_bandwidth = 1/5*delay_spread
    return coherence_bandwidth

tau_values = np.random.uniform(9e-9, 14e-9, 100)

"""### Coherence Bandwidth"""

# Constants
c = 3e8  # Speed of light (m/s)
B_start = 6e11  # Start frequency (0.06 THz)
B_end = 1e12  # End frequency (10 THz)
N_B = 100  # Number of sub-bands
S_N = 1e-20  # Power spectral density of noise (W/Hz)
P_total_dBm_values = np.linspace(-10, 30, 50)  # Transmit Power range (-10 dBm to 30 dBm)

frequencies = np.linspace(B_start, B_end, N_B)
delta_f = (B_end - B_start) / N_B
distances = [3, 4, 5, 6]
C_total_values = {d: [] for d in distances}
coherence_bandwidths = []
i = 0
for d in distances:
  B_c_los = []
  for f in frequencies:
    h_i_values = []
    for t_los in tau_values:
      h_i_values.append(np.abs(H_LoS(f, d, t_los))**2 + np.abs(H_Sca(f, rho_0=0.1, L=0.05, F=0.9, lx=0.1, ly=0.1, g=0.5, theta_1=np.pi/4))**2 + np.abs(H_Dif(f, d1, d2, h_d))**2)
    B_c_los.append(coherence_bandwidth(h_i_values, tau_values))
  coherence_bandwidths.append(B_c_los)


# Plot Coherence Bandwidth vs Frequency
plt.figure(figsize=(8, 5))
for i in range(0, 4):
  plt.plot(frequencies/1e12, coherence_bandwidths[i], label=f"LoS, Distance {distances[i]}")

plt.xlabel("Frequency (THz)")
plt.ylabel("Coherence Bandwidth (GHz)")
plt.title("Coherence Bandwidth vs Frequency for Different Distances")
plt.legend()
plt.grid()
plt.show()

"""### Channel Capacity"""

f = 6e11

for d in distances:
  for P_total_dBm in P_total_dBm_values:
    P_total = 10**(P_total_dBm / 10) / 1000  # Convert dBm to Watts
    P_i = P_total / N_B  # Equal Power Allocation (EPA)

    # Compute channel capacity per sub-band
    C_i = delta_f * (np.log2(1 + (((np.abs(H_LoS(f, d*1e-5, t_los))**2 +
                        np.abs(H_Sca(f, rho_0=0.1, L=0.05, F=0.9, lx=0.1, ly=0.1, g=0.5, theta_1=np.pi/4))**2 +
                        np.abs(H_Dif(f, d1, d2, h_d))**2)) * P_i) / (delta_f * S_N)))
    C_total = np.sum(C_i) / 1e9  # Convert to Gbps
    C_total_values[d].append(C_total)

# Plot Channel Capacity vs Transmit Power for Different Distances
plt.figure(figsize=(8, 5))
for d in distances:
    plt.plot(P_total_dBm_values, C_total_values[d], marker='o', label=f"Distance {d}")

plt.xlabel("Transmit Power (dBm)")
plt.ylabel("Total Channel Capacity (Gbps)")
plt.title("THz Wideband Channel Capacity vs Transmit Power")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm

np.random.seed(42)

# System parameters
N_t = 2 # Number of base station
K_total = 16 # Number of aircrafts at a time between two base stations
K_sched = 8 # Number of users allowed at a time
N_r = 8 # Number of receving antennas on the aircraft
SNR_dB = np.arange(-10, 30, 5)
SNR = 10**(SNR_dB / 10)
num_symbols = 1000

# Channel model
def generate_channel(N_r, N_t):
    return (np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t)) / np.sqrt(2) # Gaussian + LoS channel

# Precoding and combining
def zero_forcing_precoder(H):
    return H.conj().T @ inv(H @ H.conj().T) # Pseudo-inverse to ensure orthogonality

def mmse_combiner(H, W, noise_power):
    return inv(H @ W @ W.conj().T @ H.conj().T + noise_power * np.eye(N_r)) @ H @ W

def water_filling(gains, P_total, noise_power):
    K = len(gains)
    gains = np.array(gains)
    mu = 0
    for _ in range(100):
        mu = (P_total + np.sum(noise_power / gains)) / K
        power_alloc = np.maximum(mu - noise_power / gains, 0)
        if np.abs(np.sum(power_alloc) - P_total) < 1e-6:
            break
    return power_alloc

# QPSK modulation
def qpsk_mod(bits):
    return (1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2])

def qpsk_demod(symbols):
    bits = np.zeros(2 * len(symbols), dtype=int)
    bits[0::2] = (np.real(symbols) < 0).astype(int)
    bits[1::2] = (np.imag(symbols) < 0).astype(int)
    return bits

# User scheduling
def user_scheduling(H_all, K_sched):
    norms = [norm(H, 'fro') for H in H_all] # Square and add all elements of a matrix, square root that value and divide each element by that value
    selected_indices = np.argsort(norms)[-K_sched:]
    return selected_indices

# Generate fixed channels and precoders
H_all_users = [generate_channel(N_r, N_t) for _ in range(K_total)]
scheduled_indices = user_scheduling(H_all_users, K_sched)
H_sched = [H_all_users[i] for i in scheduled_indices]
H_stack = np.vstack(H_sched)

# Precoder
W = zero_forcing_precoder(H_stack)
W /= norm(W, 'fro')

# Gains remain constant
gains = [norm(H_sched[k] @ W[:, k])**2 for k in range(K_sched)]

# Main loop
sum_rates = []
ber_results = []

for snr_db in SNR_dB:
    snr = 10 ** (snr_db / 10)
    noise_power = 1
    tx_power = snr * noise_power

    # Power allocation stays per SNR
    power_alloc = water_filling(gains, tx_power, noise_power)

    rate = 0
    total_errors = 0
    total_bits = 0

    for k in range(K_sched):
        Hk = H_sched[k]
        Wk = W[:, k:k+1]
        Vk = mmse_combiner(Hk, Wk, noise_power)

        bits_tx = np.random.randint(0, 2, 2 * num_symbols)
        symbols = qpsk_mod(bits_tx)

        tx_signal = Wk @ (np.sqrt(power_alloc[k]) * symbols[np.newaxis, :])
        noise = np.sqrt(noise_power/2) * (np.random.randn(N_r, num_symbols) + 1j * np.random.randn(N_r, num_symbols))
        rx_signal = Hk @ tx_signal + noise
        yk = Vk.conj().T @ rx_signal

        bits_rx = qpsk_demod(yk.flatten())
        errors = np.sum(bits_rx != bits_tx)
        total_errors += errors
        total_bits += len(bits_tx)

        signal_power = np.abs(Vk.conj().T @ Hk @ Wk)**2 * power_alloc[k]
        interference = 0
        for j in range(K_sched):
            if j != k:
                Wj = W[:, j:j+1]
                interference += np.abs(Vk.conj().T @ Hk @ Wj)**2 * power_alloc[j]
        SINR = signal_power / (interference + noise_power)
        rate += np.log2(1 + SINR.real)

    sum_rates.append(rate)
    ber_results.append(total_errors / total_bits)

# Convert results to arrays
sum_rates = np.array(sum_rates).flatten()
ber_results = np.array(ber_results).flatten()

# Plotting
plt.figure()
plt.plot(SNR_dB, sum_rates, marker='o', label='Sum Rate')
plt.xlabel('SNR (dB)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Massive MU-MIMO with Fixed Channel')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.semilogy(SNR_dB, ber_results, marker='s', label='BER')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER with Fixed Channel and User Scheduling')
plt.grid(True, which='both')
plt.legend()
plt.show()

# === 1. Sum Rate for Different N_t, N_r, K_sched ===
N_t_list = [2, 4, 8]
N_r_list = [4, 8, 16]
K_sched_list = [4, 8, 12]
results = {}

for N_t in N_t_list:
    for N_r in N_r_list:
        for K_sched in K_sched_list:
            K_total = max(2 * K_sched, 16)
            H_all_users = [generate_channel(N_r, N_t) for _ in range(K_total)]
            selected_indices = user_scheduling(H_all_users, K_sched)
            H_sched = [H_all_users[i] for i in selected_indices]
            H_stack = np.vstack(H_sched)
            W = zero_forcing_precoder(H_stack)
            W /= norm(W, 'fro')
            gains = [norm(H_sched[k] @ W[:, k])**2 for k in range(K_sched)]
            power_alloc = water_filling(gains, tx_power, noise_power)

            rate = 0
            for k in range(K_sched):
                Hk = H_sched[k]
                Wk = W[:, k:k+1]
                Vk = mmse_combiner(Hk, Wk, noise_power)
                signal_power = np.abs(Vk.conj().T @ Hk @ Wk)**2 * power_alloc[k]
                interference = 0
                for j in range(K_sched):
                    if j != k:
                        Wj = W[:, j:j+1]
                        interference += np.abs(Vk.conj().T @ Hk @ Wj)**2 * power_alloc[j]
                SINR = signal_power / (interference + noise_power)
                rate += np.log2(1 + SINR.real)
            results[(N_t, N_r, K_sched)] = rate

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = [r[0] for r in results]
ys = [r[1] for r in results]
zs = [r[2] for r in results]
rates = [results[r] for r in results]
sc = ax.scatter(xs, ys, zs, c=rates, cmap='viridis', s=60)
ax.set_xlabel('N_t')
ax.set_ylabel('N_r')
ax.set_zlabel('K_sched')
fig.colorbar(sc, label='Sum Rate (bps/Hz)')
plt.title(f'Sum Rate vs N_t, N_r, K_sched at 30 dB')
plt.show()

# === 2. SINR per User vs SNR ===
N_t = 4
N_r = 8
K_total = 16
K_sched = 8
H_all_users = [generate_channel(N_r, N_t) for _ in range(K_total)]
scheduled_indices = user_scheduling(H_all_users, K_sched)
H_sched = [H_all_users[i] for i in scheduled_indices]
H_stack = np.vstack(H_sched)
W = zero_forcing_precoder(H_stack)
W /= norm(W, 'fro')
gains = [norm(H_sched[k] @ W[:, k])**2 for k in range(K_sched)]

sinr_per_user = []

for snr_db in SNR_dB:
    snr = 10 ** (snr_db / 10)
    tx_power = snr * noise_power
    power_alloc = water_filling(gains, tx_power, noise_power)

    user_sinrs = []
    for k in range(K_sched):
        Hk = H_sched[k]
        Wk = W[:, k:k+1]
        Vk = mmse_combiner(Hk, Wk, noise_power)
        signal_power = np.abs(Vk.conj().T @ Hk @ Wk)**2 * power_alloc[k]
        interference = 0
        for j in range(K_sched):
            if j != k:
                Wj = W[:, j:j+1]
                interference += np.abs(Vk.conj().T @ Hk @ Wj)**2 * power_alloc[j]
        SINR = signal_power / (interference + noise_power)
        user_sinrs.append(SINR.real)
    sinr_per_user.append(user_sinrs)

sinr_per_user = np.array(sinr_per_user)

plt.figure()
for k in range(K_sched):
    plt.plot(SNR_dB, sinr_per_user[:, k].squeeze(), marker='o', label=f'User {k+1}')
plt.xlabel('SNR (dB)')
plt.ylabel('SINR (linear)')
plt.title('SINR per User vs SNR')
plt.grid(True)
plt.legend()
plt.show()

# === 3. Water-Filling Power Allocation ===
plt.figure()
plt.plot(range(len(gains)), gains, marker='o', label='Channel Gains')
plt.plot(range(len(gains)), water_filling(gains, tx_power, noise_power), marker='x', label='Power Allocated')
plt.xlabel('User Index')
plt.ylabel('Value')
plt.title(f'Water-Filling Power Allocation at 30 dB')
plt.grid(True)
plt.legend()
plt.show()

# === 4. User Scheduling Frobenius Norm ===
fro_norms = [norm(H, 'fro') for H in H_all_users]
selected = np.zeros(K_total)
selected[scheduled_indices] = 1

plt.figure()
plt.bar(range(K_total), fro_norms, color=['red' if selected[i] else 'gray' for i in range(K_total)])
plt.xlabel('User Index')
plt.ylabel('Frobenius Norm of Channel')
plt.title('User Scheduling: Selected vs Unselected')
plt.grid(True)
plt.show()

# === 5. BER vs SNR for Different (N_r, N_t) Pairs ===
N_t_list = [2, 4]
N_r_list = [4, 8]
K_sched = 8
K_total = 2 * K_sched
num_symbols = 1000

plt.figure()

for N_t in N_t_list:
    for N_r in N_r_list:
        ber_curve = []

        H_all_users = [generate_channel(N_r, N_t) for _ in range(K_total)]
        selected_indices = user_scheduling(H_all_users, K_sched)
        H_sched = [H_all_users[i] for i in selected_indices]
        H_stack = np.vstack(H_sched)

        W = zero_forcing_precoder(H_stack)
        W /= norm(W, 'fro')
        gains = [norm(H_sched[k] @ W[:, k])**2 for k in range(K_sched)]

        for snr_db in SNR_dB:
            snr = 10 ** (snr_db / 10)
            tx_power = snr * noise_power
            power_alloc = water_filling(gains, tx_power, noise_power)

            total_errors = 0
            total_bits = 0

            for k in range(K_sched):
                Hk = H_sched[k]
                Wk = W[:, k:k+1]
                Vk = mmse_combiner(Hk, Wk, noise_power)

                bits_tx = np.random.randint(0, 2, 2 * num_symbols)
                symbols = qpsk_mod(bits_tx)

                tx_signal = Wk @ (np.sqrt(power_alloc[k]) * symbols[np.newaxis, :])
                noise = np.sqrt(noise_power / 2) * (np.random.randn(N_r, num_symbols) + 1j * np.random.randn(N_r, num_symbols))
                rx_signal = Hk @ tx_signal + noise
                yk = Vk.conj().T @ rx_signal

                bits_rx = qpsk_demod(yk.flatten())
                errors = np.sum(bits_rx != bits_tx)
                total_errors += errors
                total_bits += len(bits_tx)

            ber_curve.append(total_errors / total_bits)

        plt.semilogy(SNR_dB, ber_curve, marker='o', label=f'N_t={N_t}, N_r={N_r}')

plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR for Different (N_t, N_r) Pairs')
plt.grid(True, which='both')
plt.legend()
plt.show()
