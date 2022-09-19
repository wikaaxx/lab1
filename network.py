from cmath import sqrt
import json
from math import floor
import random
import numpy as np
import scipy.special as sp
from lightpath import Lightpath
from node import Node
from line import Line
from sig import Signal_information
from connection import Connection
from pandas import DataFrame
import matplotlib.pyplot as plt
import constants as my_cs


class Network:

    def __init__(self, path: str, channels: int = 10, transceiver: str = "fixed-rate") -> None:
        input_file = open(path)
        data = json.load(input_file)
        input_file.close()
        self.nodes = {}
        self.lines = {}
        self.channels = channels
        self.time = 0
        self.request_matrix = None

        for key, value in data.items():
            tmp_data = {
                "label": key,
                "position": (value["position"][0], value["position"][1]),
                "connected_nodes": value["connected_nodes"]
            }
            if ("transceiver" not in value):
                tmp_data['transceiver'] = transceiver
            else:
                tmp_data['transceiver'] = value['transceiver']
            self.nodes[key] = Node(tmp_data)
        for value in self.nodes.values():
            for end_l in value.connected_node:
                self.lines[value.label + end_l] = Line(value.label + end_l, sqrt(
                    (value.position[0] - self.nodes[end_l].position[0]) ** 2 + (
                                value.position[1] - self.nodes[end_l].position[1]) ** 2), channels)
                self.lines[value.label + end_l].connect(value, self.nodes[end_l])
                value.addline(self.lines[value.label + end_l], end_l)

        self.create_weighted_paths_and_route_space()

        # creating a logger
        self.logger = DataFrame(columns=["epoch time", "path", "Channel ID", "bit rate", "allocated bit rate"])

    def update_logger(self, con: Connection) -> None:
        time = self.time  # current time
        self.time += 1  # advance the time
        new_data = {"epoch time": time, "path": self.path_to_string(con.path), "Channel ID": con.channel,
                    "bit rate": con.bitRate, "allocated bit rate": con.allocated}
        self.logger.loc[time] = new_data  # add the line to the panda data frame

    def recursive_path(self, start: str, end: str, forbidden: list, all_path: list) -> list:
        if (start == end):
            all_path.append(list(forbidden))
            return
        for next_node in self.nodes[start].connected_node:
            if (next_node not in forbidden):
                forbidden.append(next_node)
                self.recursive_path(next_node, end, forbidden, all_path)
                forbidden.pop()

    def draw(self) -> None:  # draws the network on screen
        fig, graph = plt.subplots()
        graph.scatter([node.position[0] for node in self.nodes.values()],
                      [node.position[1] for node in self.nodes.values()])  # draw the nodes
        for point, name in zip([node.position for node in self.nodes.values()],
                               [label for label in self.nodes.keys()]):  # add the labels
            graph.annotate(name, xy=point, xytext=(0, -10), textcoords='offset points',
                           color='blue', ha='center', va='center')
        for line in self.lines.values():  # draw the lines
            graph.plot([node.position[0] for node in line.successive.values()],
                       [node.position[1] for node in line.successive.values()])
        plt.show()

    def find_paths(self, start: str, end: str) -> list:
        result = []
        self.recursive_path(start, end, [start], result)
        return result

    def propagate(self, signal: Signal_information) -> Signal_information:
        node = signal.update_path()
        self.nodes[node].propagate(signal)
        return signal

    def recursive_check_path(self, path: list, pos: int,
                             channel: int = 0) -> bool:  # recursively checks if the path is free or not
        if (pos == len(path) - 1):
            return True
        if (self.lines[path[pos] + path[pos + 1]].occupy(channel)):  # check channel availability
            if (not self.lines[path[pos] + path[pos + 1]].in_service):
                self.lines[path[pos] + path[pos + 1]].free(channel)
                return False
            if (self.recursive_check_path(path, pos + 1, channel)):
                self.last_channel = channel
                return True
            self.lines[path[pos] + path[pos + 1]].free(channel)
        if (pos == 0 and channel != self.channels - 1):
            return self.recursive_check_path(path, 0, channel + 1)
        else:
            return False

    def find_best_snr(self, begin: str, end: str) -> list:
        best = []
        best_snr = -1.0
        best_channel = -1
        for path in self.find_paths(begin, end):
            sig = self.propagate(Signal_information(1e-3, path))
            if ((sig.get_signal_noise_ration().real > best_snr.real or best_snr == -1) and self.recursive_check_path(
                    path, 0)):
                best_snr = sig.get_signal_noise_ration()
                if (len(best) != 0):
                    for i in range(0, len(best) - 1):
                        self.lines[best[i] + best[i + 1]].free(best_channel)
                best = list(path)
                best_channel = self.last_channel
        self.last_channel = best_channel
        return best

    def find_best_latency(self, begin: str, end: str) -> list:
        best = []
        best_snr = -1.0
        best_channel = -1
        for path in self.find_paths(begin, end):
            sig = self.propagate(Signal_information(1e-3, path))
            if ((float(sig.latency.real) < best_snr or best_snr == -1) and self.recursive_check_path(path, 0)):
                best_snr = sig.latency.real
                if (len(best) != 0):
                    for i in range(0, len(best) - 1):
                        self.lines[best[i] + best[i + 1]].free(best_channel)
                best_channel = self.last_channel
                best = list(path)
        self.last_channel = best_channel
        return best

    def stream(self, cons: list, to_use=True):  # testes all the possible connections
        for con in cons:
            if (to_use):  # depending on the best path selected I use the appropriate function
                path = self.find_best_latency(con.input, con.output)
                con.setChannel(
                    self.last_channel)  # set the channel of the connection to the last one check for inside the function
            else:
                path = self.find_best_snr(con.input, con.output)
                con.setChannel(
                    self.last_channel)  # set the channel of the connection to the last one check for inside the function

            if (len(path) != 0):  # if a path was found
                con.setBitRate(self.calculate_bit_rate(Lightpath(con.signal_power, path, con.channel), self.nodes[
                    con.input].transceiver))  # calculate the bitrate using the first node technology
                if (con.bitRate > 0):  # if the GSNR requirements are met
                    sig = self.propagate(Signal_information(con.signal_power, path))
                    con.setLatency(sig.latency.real)
                    con.setSNR(sig.get_signal_noise_ration().real)
                    con.setPath(path)

                    self.update_route_space(con)  # update route space
                    self.update_logger(con)  # update logger
                else:  # if the bitrate is 0 (GSNR requirements not met)
                    for i in range(0, len(path) - 1):  # free the line
                        self.lines[path[i] + path[i + 1]].free(con.channel)
                    con.setLatency(None)  # and reject the connection
                    con.setSNR(0.0)
            else:  # if no path is found reject the connection
                con.setLatency(None)
                con.setSNR(0.0)

    def update_route_space(self, con: Connection) -> None:  # updates the routed space to allow for the connection
        self.route_space.loc[self.path_to_string(con.path), con.channel] = False  # first here
        subPaths = []
        self.recursive_generate_sub_paths(con.path, subPaths)
        for path in subPaths:  # and then in all the subpath possible
            self.route_space.loc[self.path_to_string(path), con.channel] = False


    def recursive_generate_sub_paths(self, path: list, total: list) -> None:
        if (len(path) == 2):
            return
        path = list(path)
        tmp = path[-1]
        del path[-1]
        total.append(list(path))
        self.recursive_generate_sub_paths(path, total)
        path.append(tmp)
        tmp = path[0]
        del path[0]
        total.append(list(path))
        self.recursive_generate_sub_paths(path, total)
        return

    def path_to_string(self, path) -> str:  # given a list of nodes it turns it into a string
        tmp_s = ""
        for node in path:
            tmp_s += node
            if (node != path[-1]):
                tmp_s += "->"
        return tmp_s

    def create_weighted_paths_and_route_space(self) -> None:  # creates the weighted path and the route space
        nodes = list(self.nodes.keys())
        labels_d = []
        data = []
        for node_s in nodes:
            for node_e in nodes:
                if (node_s != node_e):
                    paths = self.find_paths(node_s, node_e)
                    for path in paths:
                        sig = self.propagate(Signal_information(1e-3, path))
                        labels_d.append(self.path_to_string(path))
                        data.append([self.path_to_string(path), sig.noise_power.real, sig.latency.real,
                                     sig.get_signal_noise_ration().real])
        self.weighted_paths = DataFrame(data, columns=['label', 'noise', 'latency', 'snr'], index=labels_d)
        data = []
        for label in labels_d:
            tmp = []
            for i in range(self.channels):
                tmp.append(True)
            data.append(tmp)
        self.route_space = DataFrame(data, index=labels_d, columns=list(range(0, self.channels)))

    def calculate_bit_rate(self, lightPath: Lightpath, strategy):
        snr = 10 ** (self.weighted_paths.loc[self.path_to_string(lightPath.path), 'snr'] / 10.0)
        return self.calculate_bit_rate_actual(snr, strategy, lightPath.Rs)

    def calculate_bit_rate_actual(self, snr, strategy,
                                  rs=my_cs.RS):  # depending on the stratefy calculates the speed depending on the snr
        if (strategy == 'fixed-rate'):
            if (snr >= 2 * sp.erfcinv(2 * my_cs.BERT) ** 2 * rs / my_cs.BN):
                return 100e9
            else:
                return 0
        elif (strategy == 'flex-rate'):
            if (snr <= 2 * sp.erfcinv(2 * my_cs.BERT) ** 2 * rs / my_cs.BN):
                return 0
            elif (snr <= 14.0 / 3.0 * sp.erfcinv(3.0 / 2.0 * my_cs.BERT) ** 2 * rs / my_cs.BN):
                return 100e9
            elif (snr <= 10.0 * sp.erfcinv(8.0 / 3.0 * my_cs.BERT) ** 2 * rs / my_cs.BN):
                return 200e9
            else:
                return 400e9
        elif (strategy == 'shannon'):
            return 2 * rs * np.log2(1.0 + snr * rs / my_cs.BN)
        else:
            return 0

    def manageTrafficMatrix(self, Tm) -> None:
        if (self.request_matrix is None):
            self.request_matrix = np.matrix(Tm)

        refused = 0
        while ((Tm != np.zeros((len(self.nodes), len(self.nodes)))).any()):  # while there is at least some request left
            nodes = [0, 0]
            while (Tm[nodes[0], int(nodes[1])] <= 0):  # select random nodes
                nodes[0] = random.randint(0, len(self.nodes) - 1)
                nodes[1] = random.randint(0, len(self.nodes) - 1)

            # convert to the string name for creating the connection

            tmp = [label for label in self.nodes.keys()]

            begin = tmp[nodes[0]]
            end = tmp[nodes[1]]

            # creating the connection
            con = Connection(begin, end, 1e-3)

            # streaming until the required traffic is allocated
            self.stream([con], False)
            while (con.bitRate < Tm[nodes[0], nodes[1]] and con.bitRate != 0):
                Tm[nodes[0], nodes[1]] -= con.bitRate
                con.allocate(con.bitRate)
                self.stream([con], False)
            if (Tm[nodes[0], nodes[1]] > 0 and con.bitRate != 0):
                con.allocate(Tm[nodes[0], nodes[1]])
                Tm[nodes[0], nodes[1]] = 0
            if (con.bitRate == 0):  # if the connection was refused register it
                refused += 1

            if (refused > 100):  # if it is no longer possible to allocate a connection terminate
                return

    def strong_failure(self, line: str) -> None:
        self.lines[line].beBrocken()

    def traffic_recovery(self) -> None:
        Tm = np.zeros((len(self.nodes), len(self.nodes)))
        for t in range(self.time):
            row = self.logger.loc[self.logger["epoch time"] == t]

            if (len(row) == 0):  # if this row was already damaged and recovered I skip it
                continue

            path = str(row["path"].iloc[0]).split("->")  # from string to path list

            # now confirming that the path is still valid
            result = True
            for i in range(len(path) - 1):
                if (not self.lines[path[i] + path[i + 1]].in_service):  # if a line is broken I signal it
                    result = False

            if (not result):  # if there is a mismatch I must correct it
                # first I free the channel and the route space
                for i in range(len(path) - 1):
                    self.lines[path[i] + path[i + 1]].free(row["Channel ID"].iloc[0])  # free that channel

                # now for the route space
                self.route_space.loc[self.path_to_string(path), row["Channel ID"].iloc[0]] = True

                subPaths = []
                self.recursive_generate_sub_paths(path, subPaths)
                for subPath in subPaths:
                    self.route_space.loc[self.path_to_string(subPath), row["Channel ID"].iloc[0]] = True

                # now creating the new connection to satisfy the same bitrate demand
                labels = [label for label in self.nodes.keys()]

                Tm[labels.index(path[0]), labels.index(path[-1])] += row["allocated bit rate"].iloc[
                    0]  # same bitrate demand as previously allocated

                self.logger.drop(row.index, inplace=True)  # remove the line after it has been fixed

        self.manageTrafficMatrix(Tm)  # reallocate all the traffic

    def total_allocated_capacity(self):
        total = 0.0
        brocken_lines = 0
        for line in self.lines.values():
            for channel in line.state:
                if (not channel):
                    total += 1
            if (not line.in_service):
                brocken_lines += 1

        return total / (self.channels * (len(self.lines) - brocken_lines)) * 100

    def average_snr(self) -> float:
        total = 0.0
        for t in range(self.time):
            row = self.logger.loc[self.logger["epoch time"] == t]
            if (len(row) == 0):
                continue
            total += self.weighted_paths.loc[row["path"].iloc[0], "snr"]
        return total / len(self.logger)

    def average_latency(self) -> float:
        total = 0.0
        for t in range(self.time):
            row = self.logger.loc[self.logger["epoch time"] == t]
            if (len(row) == 0):
                continue
            total += self.weighted_paths.loc[row["path"].iloc[0], "latency"]
        return total / len(self.logger)


def calculate_average(elems: list) -> float:
    result = 0.0
    for elem in elems:
        result += elem
    return result / len(elems)


def count_false(status: list) -> int:
    result = 0
    for s in status:
        if not s:
            result += 1
    return result


