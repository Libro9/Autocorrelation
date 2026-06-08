import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import constants
from math import *
from pylab import *
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

c = constants.c
GVD_air = 20e-30 # s^2/m, group velocity dispersion of air
GVD_glass_800mm = 36000e-30 # s^2/m, group velocity dispersion of glass at 800 nm
TOD_air = 18e-45 # s^3/m, third order dispersion of air
TOD_glass_800mm = 27000e-45 # s^3/m, third order dispersion of glass at 800 nm

class LaserPulse:
    """
    Laser pulse reconstructed from measured spectrum.
    """

    def __init__(self,
                 spectrum,
                 N=2**19,
                 threshold=0.01,
                 dt=0.05e-15,
                 absolute_phase=0.0,
                 group_delay=0.0e-15,
                 group_delay_dispersion=0.0,
                 third_order_dispersion=0.0):

        # ----------------------------------------------------
        # Basic parameters
        # ----------------------------------------------------
        self.spectrum = spectrum
        self.N = N
        self.threshold = threshold
        self.dt = dt
        self.absolute_phase = absolute_phase
        self.group_delay = group_delay
        self.group_delay_dispersion = group_delay_dispersion
        self.third_order_dispersion = third_order_dispersion
        self.fs = 1 / dt

        spectrum_lambda = spectrum[np.argsort(spectrum[:, 0])]
        freq_data = c / spectrum_lambda[:, 0]
        amp_data = spectrum_lambda[:, 1]
        idx = np.argsort(freq_data)
        spectrum = np.column_stack((freq_data[idx], amp_data[idx]))

        self.freq_data = spectrum[:, 0]
        self.amp_data = spectrum[:, 1]

        # ----------------------------------------------------
        # Background subtraction + thresholding
        # ----------------------------------------------------

        n = spectrum.shape[0]

        background = np.hstack((self.amp_data[:int(0.01 * n)],
                                self.amp_data[-int(0.01 * n):])).mean()

        self.amp_data -= background

        max_amp = np.max(self.amp_data)

        valid_idx = self.amp_data >= threshold * max_amp

        self.freq_valid = self.freq_data[valid_idx]
        self.amp_valid = self.amp_data[valid_idx]

        # ----------------------------------------------------
        # Pulse spectral properties
        # ----------------------------------------------------

        self.central_frequency = np.mean(self.freq_valid)

        self.bandwidth = self.freq_valid.max() - self.freq_valid.min()

        self.bandwidth *= 3

        # ----------------------------------------------------
        # FFT frequency bins
        # ----------------------------------------------------

        self.f_bins = np.fft.fftfreq(self.N, d=self.dt)

        # ----------------------------------------------------
        # Interpolate measured spectrum
        # onto FFT bins
        # ----------------------------------------------------

        self.E_f = np.interp(self.f_bins,
                             self.freq_valid,
                             self.amp_valid,
                             left=0,
                             right=0)

        self.E_f = self.E_f.astype(complex)

        # ----------------------------------------------------
        # Spectral phase
        # ----------------------------------------------------

        phi_0 = absolute_phase

        phi_1 = group_delay * 2 * np.pi

        phi_2 = group_delay_dispersion * (2 * np.pi)**2

        phi_3 = third_order_dispersion * (2 * np.pi)**3
        
        self.spectral_phase = (phi_0
                            + phi_1 * self.f_bins
                            + 0.5 * phi_2 * self.f_bins**2
                            + (1.0 / 6.0) * phi_3 * self.f_bins**3)

        self.E_f *= np.exp(1j * self.spectral_phase)

        # self.E_f = 0.5 * (self.E_f + np.conj(self.E_f[::-1])) #Hermition symmetry

    def plot_spectrum(self):

        plt.figure(figsize=(12, 4))

        plt.scatter(self.f_bins, np.abs(self.E_f), s=0.5)

        plt.xlabel("Frequency / Hz")
        plt.ylabel("Amplitude (arb. units)")
        # plt.xlim(3.25e14, 4.25e14)
        plt.xlim(2.5e14, 5.5e14)
        plt.title("Pulse spectrum")

        plt.grid(True)
        plt.savefig("Spectrum.png")
        plt.show()

        return self.f_bins, self.E_f

    def field(self):

        ### Computes the inverse FFT to get the time-domain field
        ### returns the time array and the electric field in the time domain
        
        self.E_t = np.fft.ifft(self.E_f)

        self.t = np.arange(self.N) * self.dt
        self.t_shift = (np.arange(self.N) - self.N // 2) * self.dt

        self.E_t_shift = np.fft.fftshift(self.E_t)
        self.E_f_shift = np.fft.fftshift(self.E_f)

        # define a time t0 where the pulse is
        peak_idx = np.argmax(np.abs(self.E_t_shift))
        self.t0 = self.t_shift[peak_idx]
        # shift = self.N // 2 - peak_idx
        # self.E_t_shift = np.roll(self.E_t_shift, shift)

        return self.t_shift, self.E_t_shift, self.t0
    
    def plot_field(self, time_window=100e-15):

        t, E, t0= self.field()

        plt.figure(figsize=(10, 6))

        plt.plot(t * 1e15,
                np.real(E) / np.max(np.abs(np.real(E))))

        plt.xlabel("Time (fs)")
        plt.ylabel("Electric field")
        plt.xlim((t0 - time_window) * 1e15, (t0 + time_window) * 1e15)
        plt.grid(True)

        plt.savefig("Field.png")
        plt.show()
    
    def autocorrelate(self, max_delay):

        #Performs autocorrelation by calculating FFT once and shifting in the time domain

        t, E, t0 = self.field()
        
        
        #set integration limits to 3x max delay to ensure we capture the full autocorrelation signal
        int_mask = (t >= t0 - 8*max_delay) & (t <= t0 + 8*max_delay)

        t = t[int_mask]
        E = E[int_mask]
        
        signal = []
        delays = np.linspace(-max_delay, max_delay, 1000)

        for d in delays:
            # shift field using interpolation (safe for non-integer shifts)
            E_delayed = np.interp(t - d, t, E, left=0, right=0)
            
            I = np.abs(E + E_delayed)**4   # intensity of SHG
            integration = np.trapezoid(I, t) # integrate over time to get autocorrelation signal at this delay
            signal.append(integration)

        signal = np.array(signal)
        signal /= signal[0]
        signal -= 1.0

        plt.plot(delays * 1e15, signal)
        plt.xlabel("Delay (fs)")
        plt.ylabel("Autocorrelation Signal")
        plt.title("Interferometric Autocorrelation")
        plt.grid(True)
        plt.savefig("Numerical_Autocorrelation1.png")
        plt.show()

        return delays, signal

    def pulse2(self, delay=0.0, added_GDD=0.0, added_TOD=0.0, plot=False):
        # Defines a modified version of the pulse that has taken the other arm of the AC,
        # identical to the original pulse but with some added delay, GDD and TOD

        pulse2 = LaserPulse(self.spectrum,
                                   N=self.N,
                                   threshold=self.threshold,
                                   dt=self.dt,
                                   absolute_phase=self.absolute_phase,
                                   group_delay=self.group_delay + delay,
                                   group_delay_dispersion=self.group_delay_dispersion + added_GDD,
                                   third_order_dispersion=self.third_order_dispersion + added_TOD)
        
        if plot:
            t, E, t0 = self.field()
            t2, E2, t0_2 = pulse2.field()
            E2= np.interp(t2 + t0_2 - t0, t2, E2, left=0, right=0) # shift pulse2 to align with original pulse in time domain
            valid_indices = np.argwhere(np.abs(E2)**2 > 0.01 * np.max(np.abs(E2)**2))

            plt.plot(t * 1e15, np.real(E), label="Arm 1 Field")
            plt.plot(t2 * 1e15, np.real(E2), label="Arm 2 Field(shifted by {:.1f} fs)".format((t0 - t0_2) * 1e15))
            plt.plot(t * 1e15, np.abs(E)**2, label="Arm 1 Intensity")
            plt.plot(t2 * 1e15, np.abs(E2)**2, label="Arm 2 Intensity")
            plt.xlabel("Time (fs)")
            plt.ylabel("Electric field")
            plt.xlim((t[valid_indices[0]] * 1e15), (t[valid_indices[-1]] * 1e15))
            plt.legend()
            plt.grid(True)
            plt.show()

        return pulse2

    def autocorrelate2(self, max_delay):

        #Performs autocorrelation by calculating FFT for each delay, including GVD and TOD from air
        t, E, t0 = self.field()
        
        #set integration limits to 3x max delay to ensure we capture the full autocorrelation signal
        int_mask = (t >= t0 - 8*max_delay) & (t <= t0 + 8*max_delay)


        t = t[int_mask]
        E = E[int_mask]
        
        signal = []
        delays = np.linspace(-max_delay, max_delay, 1000)

        for d in delays:
            delayed_pulse = LaserPulse(self.spectrum,
                                   N=self.N,
                                   threshold=self.threshold,
                                   dt=self.dt,
                                   absolute_phase=self.absolute_phase,
                                   group_delay=self.group_delay + d,
                                   group_delay_dispersion=self.group_delay_dispersion + GVD_air * d * c, # add GVD from air for this delay
                                   third_order_dispersion=self.third_order_dispersion + TOD_air * d * c) # add TOD from air for this delay
            
            t_delayed, E_delayed, t0_delayed = delayed_pulse.field()
            t_delayed = t_delayed[int_mask]
            E_delayed = E_delayed[int_mask]
            
            I = np.abs(E + E_delayed)**4   # intensity of SHG
            integration = np.trapezoid(I, t) # integrate over time to get autocorrelation signal at this delay
            signal.append(integration)

        signal = np.array(signal)
        signal /= signal[0]
        signal -= 1.0

        plt.plot(delays * 1e15, signal)
        plt.xlabel("Delay (fs)")
        plt.ylabel("Autocorrelation Signal")
        plt.title("Interferometric Autocorrelation")
        plt.grid(True)
        plt.savefig("Numerical_Autocorrelation2.png")
        plt.show()
        
        return delays, signal
    
    def autocorrelate3(self, max_delay, num_delays=1000, added_delay=0.0, added_GDD=0.0, added_TOD=0.0, plot=False):

        #Performs autocorrelation by calculating FFT twice, once for the original pulse and once with some added delay, GDD and TOD,
        #then shifting in the time domain for each delay

        t, E, t0 = self.field()
        
        #set integration limits to 8x max delay to ensure we capture the full autocorrelation signal
        int_mask = (t >= t0 - 8*max_delay) & (t <= t0 + 8*max_delay)

        t = t[int_mask]
        E = E[int_mask]

        pulse2 = self.pulse2(added_delay, added_GDD, added_TOD)
        
        t2, E2, t0_2 = pulse2.field()
        E2 = np.interp(t2 + t0_2 - t0, t2, E2, left=0, right=0) # shift pulse2 to align with original pulse
        # int_mask_prism = (t_prism >= t0_prism - 8*max_delay) & (t_prism <= t0_prism + 8*max_delay)
        t2 = t2[int_mask]
        E2 = E2[int_mask]

        signal = []
        delays = np.linspace(-max_delay, max_delay, num_delays)

        for d in delays:
            # shift field using interpolation (safe for non-integer shifts)
            E_delayed = np.interp(t2 - d, t2, E2, left=0, right=0)
            
            I = np.abs(E + E_delayed)**4   # intensity of SHG
            integration = np.trapezoid(I, t) # integrate over time to get autocorrelation signal at this delay
            signal.append(integration)

        signal = np.array(signal)
        signal /= signal[0]
        signal -= 1.0

        if plot:

            plt.plot(delays * 1e15, signal)
            plt.xlabel("Delay (fs)")
            plt.ylabel("Autocorrelation Signal")
            plt.title("Interferometric Autocorrelation")
            plt.grid(True)
            plt.savefig("Numerical_Autocorrelation1.png")
            plt.show()

        return delays, signal
    
    def intensity_autocorrelation(self, max_delay, num_delays=1000, added_delay=0.0, added_GDD=0.0, added_TOD=0.0):

        pulse2 = self.pulse2(added_delay, added_GDD, added_TOD)

        t, E, t0 = self.field()
        
        #set integration limits to 8x max delay to ensure we capture the full autocorrelation signal
        int_mask = (t >= t0 - 8*max_delay) & (t <= t0 + 8*max_delay)

        t = t[int_mask]
        E = E[int_mask]
        
        t2, E2, t0_2 = pulse2.field()
        E2 = np.interp(t2 + t0_2 - t0, t2, E2, left=0, right=0) # shift pulse2 to align with original pulse in time domain
        t2 = t2[int_mask]
        E2 = E2[int_mask]

        signal = []
        delays = np.linspace(-max_delay, max_delay, num_delays)

        for d in delays:
            # shift field using interpolation (safe for non-integer shifts)
            E_delayed = np.interp(t2 - d, t2, E2, left=0, right=0)
            
            #I = np.abs(E)**2 * np.abs(E_delayed)**2   # intensity autocorrelation
            I = (np.abs(E)**2 + np.abs(E_delayed)**2)**2   # intensity autocorrelation
            integration = np.trapezoid(I, t) # integrate over time to get autocorrelation signal at this delay
            signal.append(integration)

        signal = np.array(signal)
        signal /= signal[0]
        #signal -= 1.0

        plt.plot(delays * 1e15, signal)
        plt.xlabel("Delay (fs)")
        plt.ylabel("Intensity Autocorrelation Signal")
        plt.title("Intensity Autocorrelation")
        plt.grid(True)
        plt.savefig("Numerical_Intensity_Autocorrelation.png")
        plt.show()

        return delays, signal
