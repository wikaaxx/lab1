from cmath import log10


class Signal_information:

    def __init__(self, signal_power, path):
        self.signal_power = signal_power
        self.noise_power = 0.0
        self.latency = 0.0
        self.path = list(path)

    def inc_sig_pow(self, power):
        self.signal_power += power

    def inc_noise_pow(self, power):
        self.noise_power += power

    def inc_latency(self, latency):
        self.latency += latency

    def update_path(self):
        if len(self.path) > 1:
            t = self.path.pop(0)
        else:
            t = self.path[0]
        return t

    def get_signal_pow(self):
        return self.signal_power

    def get_snr(self):
        return 10 * (log10(self.signal_power.real) - log10(self.noise_power.real))

    def get_latency(self):
        return self.latency

    def __str__(self):
        return str(self.get_snr()) + " - " + str(self.get_latency())
