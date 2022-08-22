
from sig import Signal_information




class Node:

    def __init__(self, input: dict):
        self.label = input["label"]
        self.position = input["position"]
        self.connected_nodes = input["connected_nodes"]
        self.successive = {}

    def get_position(self):
        return self.position

    def get_label(self):
        return self.label

    def successive(self, successive):
        self.successive = successive


    def propagate(self,sig: Signal_information):
        next =sig.path_update()
        if next in self.connected_nodes:
            self.successive[next].propagate(sig)






