import pandas as pd
from sig import Signal_information
from network import Network
import math
import json


network = Network('nodes.json')
network.connect()
node_l=network.nodes.keys()
pairs=[]
for i in node_l:
    for i2 in node_l:
        if i != i2:
            pairs.append(i+i2)
columns=['path', 'latency', 'noise', 'snr']
df=pd.DataFrame()
paths=[]
latencies=[]
noises=[]
snrs=[]

for p in pairs:
    for path in network.find_paths(p[0], p[1]):
        path_str=''
        for node in path:
            path_str+=node+'->'
        paths.append(path_str[:-2])

        sig = Signal_information(1, path)
        sig= network.propagate(sig)
        latencies.append(sig.latency)
        noises.append(sig.noise_power)
        snrs.append(10*math.log(sig.signal_power/sig.noise_power,10))

df['path'] = paths
df['latency'] = latencies
df['noise'] = noises
df['snr'] = snrs


