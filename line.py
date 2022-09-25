from math import floor
from sig import Signal_information
import numpy as np
from scipy.constants import c, h, pi, e


class Line:
    gain = 16  # dB!!!
    #noise_figure = 3  # dB!!!!!!
    noise_figure = 5 # for the third simulation
    alpha = 0.2e-3  # dB/m
    module_beta = 2.13e-26  # 1/(m * Hz^2)
    # module_beta = 0.6e-26 # for the second simulation
    gamma = 1.27e-3  # (m*W)^-1
    Rs = 32e9  # Hz
    df = 50e9  # Hz
    f = 193.414e12
    NLI = -1.0

    def __init__(self, label, length, channels: int) -> None:
        self.label = label
        self.length = length
        self.successive = {}
        self.state = []  # indicate occupancy of each ch
        self.in_service = True  # for strong failure
        for i in range(0, channels):
            self.state.append(True)
        self.n_amp = floor(self.length.real / 80e3.real) + 2  # amp every 80km + 2(booster+pre-amp)

    def connect(self, begin, end):
        self.successive["begin"] = begin
        self.successive["end"] = end

    def noise_generation(self, signal_power):
        ase = self.ase_generation()
        nli = self.nli_generation(signal_power)
        return ase + nli

    def latency_generation(self):
        return self.length / (2 / 3 * c)

    def propagate(self, signal: Signal_information):
        signal.inc_noise_pow(self.noise_generation(signal.get_signal_pow()))
        signal.inc_latency(self.latency_generation())
        self.successive["end"].propagate(signal)

    def occupy(self, channel):
        if self.state[channel]:
            self.state[channel] = False
            return True
        return False

    def free(self, channel):
        channel = int(channel)
        if not self.state[channel]:
            self.state[channel] = True

    def getFreeChannel(self):  # return first available channel or -1 if there none, and will call occupy on it
        i = 0
        while i < len(self.state) and not self.state[i]:
            i += 1
        if i == len(self.state):
            return -1
        else:
            return i

    def ase_generation(self):
        return self.n_amp * (h * 12.5e9 * 10 ** (self.noise_figure / 10) * (10 ** (self.gain / 10) - 1) * self.f)

    def nli_generation(self, signal_power):
        return signal_power ** 3 * self.calculate_nli() * 12.5e9 * 10 ** (-self.alpha * self.length / 10) * 10 ** (
                self.gain / 10)

    def optimized_launch_power(self):  # slide 31 of OLS(8)
        l = np.abs(20 * np.log10(e) / self.alpha)
        return (self.length * self.noise_figure * (h * 12.5e9 * self.f) / (
                2 * 12.5e9 * self.calculate_nli())) ** (1 / 3)

    def calculate_nli(self):

        if self.NLI == -1:  # if not yet done
            alpha = np.abs(self.alpha / (10 * np.log10(e)))
            log_arg = pi ** 2 * self.module_beta * self.Rs ** 2 * len(self.state) ** (2 * self.Rs / self.df) / (
                    2 * alpha)  # argument of log
            factor = 16 / (27 * pi) * self.gamma ** 2 / (
                    4 * alpha * self.module_beta * self.Rs ** 3)  # the other factor
            self.NLI = factor * np.log(log_arg)

        return self.NLI

    def set_in_service(self,service):
        self.in_service = service
