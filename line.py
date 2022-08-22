from typing import TYPE_CHECKING
from sig import Signal_information

if TYPE_CHECKING:
    from node import Node

class Line:
    def __init__(self,input: dict):
        self.label=input["label"]
        self.length=input["length"]
        self.successive={}

    def length(self):
        return self.length()
    def label(self):
        return self.label()

    def successive(self, successive):
        self.successive = successive

    def latency_generation(self):
        v = float((2/3)*3*10**8)
        return self.length/v

    def noise_generation(self, sig_power :float):
        return 1e-9*sig_power*self.length

    def propagate(self,sig: Signal_information):
        sig.addlatency(self.latency_generation())
        noise_power=self.noise_generation(sig.signal_power)
        sig.addnoisepower(noise_power)


