


class Signal_information:


    def __init__(self, signal_power, path):
        self.signal_power = signal_power
        self.path = list(path)
        self.latency = 0.0
        self.noise_power = 0.0

    def addsigpower(self, power):
        self.signal_power += power

    def addnoisepower(self, noise):
        self.noise_power += noise

    def getpower(self):
        return self.signal_power

    def getnoise(self):
        return self.noise_power

    def addlatency(self, latency):
        self.latency += latency

    def path_update(self):
        self.path = self.path.pop(0)



