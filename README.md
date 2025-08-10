# Channel-Modelling-and-Analysis-for-Aircraft-Communication
Objectives
- To use Python to model hypothetical channel in order to observe effects of various factors, mainly frequency
- Evaluate band capacity
- Undertake a performance analysis via metrics like SNR and BER simulation
- Comparison with other standard channel models like Rician


This project focuses on modeling and analyzing THz channels for aircraft communication and develop multi-ray propagation of the line-of-sight (LoS), reflected, scattered and diffracted paths. The channel parameters of the Terahertz spectrum such as the path gain, the wideband channel capacity, the rms delay spread and the temporal broadening effects need to be accurately investigated. 

Line-of-Sight (LoS)
The line-of-sight (LoS) transfer function describes the direct transmission path between transmitter and receiver, essential for maximising signal strength in MIMO systems. It depends on distance, frequency, and propagation delay, impacting overall channel capacity.

H_LoS = H_Spr(f, r) * np.exp(-1j * 2 *np.pi * f * t_los)

H_Spr = c / (4 * np.pi * f * r)

Reflected Signal
- Reflected signal transfer function captures the impact of surfaces reflecting signals in THz channels.
- Calculated complex refractive index,Gamma_TE, static permittivity, high frequency permittivity and other parameters

Scattering
- Scattering in THz communication occurs when signals deviate due to interactions with particles and obstacles in the environment.
- Calculated using roughness parameter, characteristic length scale, Fresnel coefficient, correlation length along x-axis and y-axis, RMS surface slope and incident zenith angle

Diffraction
- Diffraction in THz communication refers to the bending of signals around obstacles, affecting overall signal propagation.
- H_Dif = (c / (4 * np.pi * f * (d1 + d2))) *np.exp(exponent) * L

Coherence Bandwidth

- Coherence bandwidth indicate sthe range of frequencies over which the channel can be considered flat.
- Determines how frequencyselective fading can impact THz communication reliability.

Channel Capacity

- Channel capacity quantifies the maximum data rate for reliable communication within a given channel.
- Channel Capacity vs Transmit Power for different distances in km are shown

Overview of MU-MIMO System

MU-MIMO systems significantly improve data throughput, reduced latency and spectral efficiency by utilizing numerous antennas for both transmission and reception.

Zero-Forcing Precoding

Zero-Forcing (ZF) precoding alleviates multi-user interference by selectively filtering the transmitted signals to ensure each user's signal is decoded without interference. This technique mathematically adjusts the transmitted signals based on the channel state information to optimise the overall data rate in the system.

MMSE Combining

Minimum Mean Square Error (MMSE) combining minimizes the error in the estimated received signals by considering the noise power in the channel. This adaptive technique allows for efficient signal processing at the receiver end, ensuring reliable communication even in noisy environments, thus enhancing the overall system performance.

Water-Filling Power Allocation

Water-filling power allocation efficiently distributes the total available transmit power among users based on channel gains, optimizing overall system throughput. By applying this method, higher power is allocated to users with better channel conditions, enhancing performance and maximizing beneficial data rates for all users.

QPSK Modulation

Quadrature Phase Shift Keying (QPSK) is a modulation scheme that encodes data bits in four distinct phase shifts of the carrier wave. Its efficient use of bandwidth allows the transmission of multiple bits per symbol, making it ideal for highcapacity applications in MU-MIMO systems, such as aircraft communication.

User Scheduling

User scheduling is critical in MUMIMO systems, where users are selected based on their channel strengths for optimal resource utilization. By prioritizing users with favorable channel conditions, the system can improve throughput and maintain efficient communication, even undervarying conditions and user loads.


Reference Paper:

C. Han, A. O. Bicen, and I. F. Akyildiz, "Multi-Ray Channel Modeling and Wideband Characterization for Wireless Communications in the Terahertz Band," IEEE Transactions on Wireless Communications, vol. 14, Issue. 5, May 2015.
