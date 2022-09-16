import json
from node import Node
from sig import Signal_information
from line import Line
from cmath import sqrt
import random
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


class Network(object):
    def __init__(self, path):
        self.nodes={}
        self.lines={}
        node_json=json.load(open(path, 'r'))
        for node_l in node_json:
            node_dict=node_json[node_l]
            node_dict['label'] = node_l
            node = Node(node_dict)
            self.nodes[node_l] = node

            for connected_l in node_dict['connected_nodes']:
                line_dict={}
                line_l= node_l + connected_l
                line_dict['label'] = line_l
                node_pos= np.array(node_json[node_l]['position'])
                connected_pos=np.array(node_json[connected_l]['position'])
                line_dict['length']=np.sqrt(np.sum((node_pos - connected_pos)**2))
                line = Line(line_dict)
                self.lines[line_l] = line

    def nodes(self):
        return self.nodes

    def lines(self):
        return self.lines

    def connect(self):
        nodes_dictionary=self.nodes
        lines_dictionary=self.lines
        for node_l in nodes_dictionary:
            node=nodes_dictionary[node_l]
            for connected_n in node.connected_nodes:
                line_l=node_l+connected_n
                line=lines_dictionary[line_l]
                line.successive[connected_n]=nodes_dictionary[connected_n]
                node.successive[line_l]=lines_dictionary[line_l]

    def find_paths(self,label1,label2):
        cross_nodes = [key for key in self.nodes.keys() if((key!=label1)&(key!=label2))]
        cross_lines=self.lines.keys()
        inner_paths={}
        inner_paths['0'] = label1
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

    def propagate(self,signal_information):
        path = signal_information.path
        start=self.nodes[path[0]]
        propagated = start.propagate(signal_information)
        return propagated

    def draw(self):
        nodes =self.nodes
        for node_label in nodes:
            n0=nodes[node_label]
            x0=n0.position[0]
            y0=n0.position[1]
            plt.plot(x0,y0,'go',markersize=10)
            plt.text(x0+20,y0+20,node_label)
            for connected in n0.connected_nodes:
                n1=nodes[connected]
                x1=n1.position[0]
                y1=n1.position[1]
                plt.plot([x0,x1],[y0,y1],'b')
        plt.xlabel('km')
        plt.ylabel('network')
        plt.show()



