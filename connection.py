class Connection:

    def __init__(self, input: str, output: str, signal_power: float):
        self.input = input
        self.output = output
        self.signal_power = signal_power
        self.channel = -1
        self.bitRate = 0
        self.latency = 0.0
        self.snr = 0.0
        self.allocated = 0.0
        self.path = []

    def set_latency(self, latency: float):
        self.latency = latency

    def set_snr(self, snr: float):
        self.snr = snr

    def set_channel(self, channel: int):
        self.channel = channel

    def set_bit_rate(self, bitRate):
        self.bitRate = bitRate

    def set_path(self, path):
        self.path = path

    def allocate(self, bitRate):
        if self.allocated + bitRate <= self.bitRate:
            self.allocated += bitRate

