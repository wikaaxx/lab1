import json
from node import Node
from sig import Signal_information
from line import Line
from cmath import sqrt
import random
from pandas import DataFrame
import matplotlib.pyplot as plt


class Network:
    def __init__(self, path):
        self.nodes={}
        self.lines={}
        node_json=json.load()



    def get_nodes(self):
        return self.nodes
    def get_lines(self):
        return self.lines
    def connect(self):
        nodes_dictionary=self.nodes
        lines_dictionary=self.lines
        for x in nodes_dictionary.values():
            for connected in x.connected_nodes:
                line = lines_dictionary[x.label+connected]
                line.successive[connected]=nodes_dictionary[connected]
                x.successive[x.label+connected]=lines_dictionary[x.label+connected]

    def find_paths(self,label1,label2):
        cross_nodes = [key for key in self.nodes.keys() if((key!=label1)&(key!=label2))]
        cross_lines=self.lines.keys()
        inner_paths={'0':label1}
        for i in range(len(cross_nodes)+1):
            inner_paths[str(i+1)]=[]
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i+1)] += [inner_path+cross_node for cross_node in cross_nodes if ((inner_path[-1]+cross_node in cross_lines)&(cross_node not in inner_path))]

        paths=[]
        for i in range(len(cross_nodes)+1):
            for path in inner_paths[str(i)]:
                if path[-1]+label2 in cross_lines:
                    paths.append(path+label2)
        return paths

    def propagate(self,sig:Signal_information):
        node = self.nodes[sig.path[0]]
        return self.nodes[node].propagate(sig)

    def draw(self):
        nodes =self.nodes
        for node_label in nodes:
            n0=nodes[node_label]
            x0=n0.position[0]
            y0=n0.position[1]
            plt.text(x0,y0,node_label)
            for connected in n0.connected_nodes:
                n1=nodes[connected]
                x1=n1.position[0]
                y1=n1.position[1]
                plt.plot([x0,x1],[y0,y1])
        plt.xlabel('km')
        plt.ylabel('network')
        plt.show()

