from typing import TYPE_CHECKING
from sig import Signal_information

if TYPE_CHECKING:
    from node import Node

class Line(object):
    def __init__(self,line_dict):
        self.label=line_dict['label']
        self.length=line_dict['length']
        self.successive={}

    def length(self):
        return self.length
    def label(self):
        return self.label
    def successive(self):
        return self.successive

    def successive(self, successive):
        self.successive = successive

    def latency_generation(self):
        v = float((2/3)*3*10**8)
        latency = self.length/v
        return latency

    def noise_generation(self, sig_power):
        noise=sig_power / (2*self.length)
        return noise

    def propagate(self,signal_information):
        latency = self.latency_generation()
        signal_information.addlatency(latency)
        signal_power= signal_information.signal_power
        noise=self.noise_generation(signal_power)
        signal_information.addnoisepower(noise)
        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information)
        return signal_information


