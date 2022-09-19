from math import floor
from sig import Signal_information
import numpy as np
from scipy import constants as cs
import constants as my_cs



class Line:
    gain = 16  # dB
    noise_figure = 3  # dB
    # noise_figure = 5 # alternative value
    alpha = 0.2e-3  # / (10*np.log10(cs.e)) #dB/m this a 10 or a 20? To ask later
    #module_beta = 2.13e-26  # 1/(m * Hz^2)
    module_beta = 0.6e-26 # alternative value
    gamma = 1.27e-3  # 1/(m*W)
    Rs = 32e9  # Hz
    df = 50e9  # Hz
    f = 193.414e12
    NLI_coeff = -1.0

    def __init__(self, label, length, channels: int) -> None:
        self.label = label
        self.length = length
        self.successive = {}
        self.state = []
        self.in_service = True
        for i in range(0, channels):
            self.state.append(True)
        self.n_amplifiers = floor(
            self.length.real / 80e3.real) + 2  # an amplifier every 80km plus the ones at the beginning and end of the line

    def connect(self, begin, end):
        self.successive["begin"] = begin
        self.successive["end"] = end

    """
    def noise_generation(self, signal_power : float) -> float:
        return 1e-9 * signal_power * self.length
    """

    def noise_generation(self, signal_power: float) -> float:
        ase = self.ase_generation()
        nli = self.nli_generation(signal_power)
        return ase + nli

    def latency_generation(self) -> float:
        speed = 2 / 3 * cs.c  # m/s using the precise constant now
        return self.length / speed

    def propagate(self, signal: Signal_information):
        signal.inc_noise_pow(self.noise_generation(signal.get_signal_pow()))
        signal.inc_latency(self.latency_generation())
        self.successive["end"].propagate(signal)

    def occupy(self, channel: int) -> bool:
        if (self.state[channel]):
            self.state[channel] = False
            return True
        return False

    def free(self, channel: int) -> None:
        channel = int(channel)
        if (not self.state[channel]):
            self.state[channel] = True

    def getFreeChannel(
            self) -> int:  # will return the first available channel or -1 if there are none, and will call occupy on it
        i = 0
        while (i < len(self.state) and not self.occupy(i)):
            i += 1
        if (i == len(self.state)):
            return -1
        else:
            return i

    def ase_generation(self):
        ase = self.n_amplifiers * (
                    cs.h * my_cs.BN * 10 ** (self.noise_figure / 10) * (10 ** (self.gain / 10) - 1) * self.f)
        return ase

    def nli_generation(self, signal_power):
        nli = signal_power ** 3 * self.calculate_NLI_coeff() * my_cs.BN * 10 ** (
                    -self.alpha * self.length / 10) * 10 ** (
                          self.gain / 10)  # np.abs(self.alpha / (10 * np.log10(cs.e))) * 80e3
        return nli

    def optimized_launch_power(self) -> float:  # slide 31 of OLS(8)
        l = np.abs(20 * np.log10(cs.e) / self.alpha)
        return (self.length * self.noise_figure * (cs.h * my_cs.BN * self.f) / (
                    2 * my_cs.BN * self.calculate_NLI_coeff())) ** (1 / 3)

    def calculate_NLI_coeff(self) -> float:  # slide 16 of OLS(8)
        if (self.NLI_coeff == -1):  # since this number does not change do the calculation only once
            alpha = np.abs(self.alpha / (10 * np.log10(cs.e)))
            log_arg = cs.pi ** 2 * self.module_beta * self.Rs ** 2 * len(self.state) ** (2 * self.Rs / self.df) / (
                        2 * alpha)  # argument of log
            factor = 16 / (27 * cs.pi) * self.gamma ** 2 / (
                        4 * alpha * self.module_beta * self.Rs ** 3)  # the other factor
            self.NLI_coeff = factor * np.log(log_arg)
        return self.NLI_coeff

    def beBrocken(self) -> None:
        self.in_service = False