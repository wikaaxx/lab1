
from sig import Signal_information




class Node(object):

    def __init__(self, node_dict):
        self.label=node_dict['label']
        self.position=node_dict['position']
        self.connected_nodes=node_dict['connected_nodes']
        self.successive={}

    def position(self):
        return self.position

    def label(self):
        return self.label

    def successive(self, successive):
        self.successive = successive

    def successive(self):
        return self.successive

    def propagate(self,signal_information: Signal_information):
        path = signal_information.path
        if len(path)>1:
            line_l=path[:2]
            line = self.successive[line_l]
            signal_information.next()
            signal_information= line.propagate(signal_information)

        return signal_information


    def connected_nodes(self):
        return self.connected_nodes


