


class Signal_information:


    def __init__(self, signal_power: float, path):
        self.signal_power = signal_power
        self.path = list(path)
        self.latency = 0.0
        self.noise_power = 0.0

    def addsigpower(self, power: float):
        self.signal_power += power

    def addnoisepower(self, noise: float):
        self.noise_power += noise

    def addlatency(self, latency: float):
        self.latency += latency

    def path_update(self):
        self.path = self.path.pop(0)

