from typing import TYPE_CHECKING
from sig import Signal_information

if TYPE_CHECKING:
    from line import Line


class Node:

    def __init__(self, data: dict):
        self.label = data["label"]
        self.position = data["position"]
        self.connected_node = data["connected_nodes"]
        self.transceiver = data['transceiver']
        self.successive = {}

    def get_position(self):
        return self.position

    def get_connections(self) -> str:
        return self.connected_node

    def addLine(self, line, dest: str) -> None:
        self.successive[dest] = line

    def propagate(self, signal: Signal_information):
        next_node = signal.update_path()
        if (next_node in self.connected_node):
            signal.signal_power = self.successive[
                next_node].optimized_launch_power()  # set the optimized launch power of the signal
            self.successive[next_node].propagate(signal)  # for this line
