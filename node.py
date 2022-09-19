from sig import Signal_information
from math import floor
from typing import TYPE_CHECKING
from sig import Signal_information
import numpy as np
from scipy import constants as cs
import constants as my_cs

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

    def addline(self, line, dest: str):
        self.successive[dest] = line

    def propagate(self, signal: Signal_information):
        next = signal.update_path()
        if (next in self.connected_node):
            l = np.abs(20 * np.log10(cs.e) / self.successive[next].alpha)
            signal.signal_power = (self.successive[next].length * self.successive[next].noise_figure * (cs.h * my_cs.BN * self.successive[next].f) / (
                    2 * my_cs.BN * self.successive[next].calculate_NLI_coeff())) ** (1 / 3) #optimal launch power computation for line
            self.successive[next].propagate(signal)


