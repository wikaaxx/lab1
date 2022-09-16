class Signal_information(object):

    def __init__(self, signal_power, path):
        self.signal_power = signal_power
        self.path = path
        self.latency = 0
        self.noise_power = 0

    def addsigpower(self, power):
        self.signal_power += power

    def addnoisepower(self, noise):
        self.noise_power += noise

    def getpower(self):
        return self.signal_power

    def getnoise(self):
        return self.noise_power

    def latency(self):
        return self.latency

    def addlatency(self, latency):
        self.latency += latency

    def next(self):
        self.path = self.path[1:]

    def latency(self, latency):
        self.latency = latency

    def power(self):
        return self.signal_power

    def noise_power(self):
        return self.noise_power

    def path(self, path):
        self.path = path

    def path(self):
        return self.path