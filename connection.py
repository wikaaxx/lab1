class Connection:

    def __init__(self, input: str, output: str, signal_power: float) -> None:
        self.input = input
        self.output = output
        self.signal_power = signal_power
        self.channel = -1
        self.bitRate = 0
        self.latency = 0.0
        self.snr = 0.0
        self.allocated = 0.0
        self.path = []

    def setLatency(self, latency: float) -> None:
        self.latency = latency

    def setSNR(self, snr: float) -> None:
        self.snr = snr

    def setChannel(self, channel: int) -> None:
        self.channel = channel

    def setBitRate(self, bitRate) -> None:
        self.bitRate = bitRate

    def setPath(self, path) -> None:
        self.path = path

    def allocate(self, bitRate) -> None:
        if (self.allocated + bitRate <= self.bitRate):
            self.allocated += bitRate

