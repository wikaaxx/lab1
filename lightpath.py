from sig import Signal_information


class Lightpath(Signal_information):
    Rs = 32e9  # Hz
    df = 50e9  # Hz

    def __init__(self, signal_power: float, path: list[str], channel: int) -> None:
        super().__init__(signal_power, path)
        self.channel = channel

    def getChannel(self) -> int:
        return self.channel

    def setChannel(self, channel: int) -> None:
        self.channel = channel