from sig import Signal_information
from scipy.constants import c, h, pi, e

class Node:



    def __init__(self, node):
        self.label = node["label"]
        self.position = node["position"]
        self.connected_node = node["connected_nodes"]
        self.transceiver = node['transceiver']
        self.successive = {}

    def label(self):
        return self.label

    def position(self):
        return self.position

    def connected_node(self):
        return self.connected_node

    def transceiver(self):
        return self.transceiver

    def addline(self, line, dest):
        self.successive[dest] = line

    def propagate(self, signal: Signal_information):
        next = signal.update_path()
        if next in self.connected_node:  # update optimal launch power
            signal.signal_power = (self.successive[next].length * self.successive[next].noise_figure * (h * 12.5e9 * self.successive[next].f) / (
                    2 * 12.5e9 * self.successive[next].calculate_nli())) ** (1 / 3)   #12.5e9 Bn
            self.successive[next].propagate(signal)


