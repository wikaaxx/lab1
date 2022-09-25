from math import floor

from network import *

network = Network("274328.json", 10)
network.draw()
nodes = list(network.nodes.keys())
cons = []
for i in range(1, 80):
    s = floor(random.uniform(0, len(nodes)))
    e = floor(random.uniform(0, len(nodes)))
    while e == s:
        e = floor(random.uniform(0, len(nodes)))
    cons.append(Connection(nodes[s], nodes[e], 1e-3))
network.stream(cons)

db_a = []
speed_fixed_a = []
speed_flex_a = []
speed_shannon_a = []
for k in range(-20, 40):
    db_a.append(k)
    snr = 10 ** (k / 10.0)
    speed_fixed_a.append(network.calculate_bit_rate_actual(snr, "fixed-rate") / 1e9)
    speed_flex_a.append(network.calculate_bit_rate_actual(snr, "flex-rate") / 1e9)
    speed_shannon_a.append(network.calculate_bit_rate_actual(snr, "shannon") / 1e9)

# speed curves
plt.plot(db_a, speed_fixed_a)
plt.xlabel("GSNR [dB]")
plt.ylabel("Fixed Speed [Gb/s]")
plt.show()

plt.plot(db_a, speed_flex_a)
plt.xlabel("GSNR [dB]")
plt.ylabel("Flex Speed [Gb/s]")
plt.show()

plt.plot(db_a, speed_shannon_a)
plt.xlabel("GSNR [dB]")
plt.ylabel("Shannon Speed [Gb/s]")
plt.show()

fixedRateNet = Network("274328.json", 10, "fixed-rate")
flexRateNet = Network("274328.json", 10, "flex-rate")
shannonNet = Network("274328.json", 10, "shannon")

nodes = list(fixedRateNet.nodes.keys())
fixedCons = []
for i in range(0, 100):
    s = floor(random.uniform(0, len(nodes)))
    e = floor(random.uniform(0, len(nodes)))
    while e == s:
        e = floor(random.uniform(0, len(nodes)))
    fixedCons.append(Connection(nodes[s], nodes[e], 1e-3))

flexCons = []
for i in range(0, 100):
    s = floor(random.uniform(0, len(nodes)))
    e = floor(random.uniform(0, len(nodes)))
    while e == s:
        e = floor(random.uniform(0, len(nodes)))
    flexCons.append(Connection(nodes[s], nodes[e], 1e-3))

shannonCons = []
for i in range(0, 100):
    s = floor(random.uniform(0, len(nodes)))
    e = floor(random.uniform(0, len(nodes)))
    while e == s:
        e = floor(random.uniform(0, len(nodes)))
    shannonCons.append(Connection(nodes[s], nodes[e], 1e-3))

fixedRateNet.stream(fixedCons, False)
flexRateNet.stream(flexCons, False)
shannonNet.stream(shannonCons, False)

plt.hist(list(map(lambda x: x.bitRate / 1e9, fixedCons)), bins=10)
plt.xlabel("bit Rate Gb/s")
plt.title("Fixed-rate transceiver")
plt.show()

plt.hist(list(map(lambda x: x.bitRate / 1e9, flexCons)), bins=4)
plt.xlabel("bit Rate Gb/s")
plt.title("Flex-rate transceiver")
plt.show()

plt.hist(list(map(lambda x: x.bitRate / 1e9, shannonCons)), bins=20)
plt.xlabel("bit Rate Gb/s")
plt.title("shannon transceiver")
plt.show()

# only maintain accepted connections
fixedCons = [con for con in fixedCons if con.bitRate > 0]
flexCons = [con for con in flexCons if con.bitRate > 0]
shannonCons = [con for con in shannonCons if con.bitRate > 0]

# print average speeds
print("Avg speed fixed: " + str(average([con.bitRate for con in fixedCons])))
print("Avg speed flex: " + str(average([con.bitRate for con in flexCons])))
print("Avg speed shannon: " + str(average([con.bitRate for con in shannonCons])))

print("Avg snr fixed: " + str(average([con.snr for con in fixedCons])))
print("Avg snr flex: " + str(average([con.snr for con in flexCons])))
print("Avg snr shannon: " + str(average([con.snr for con in shannonCons])))

print("Avg latency fixed: " + str(average([con.latency for con in fixedCons])))
print("Avg latency flex: " + str(average([con.latency for con in flexCons])))
print("Avg latency shannon: " + str(average([con.latency for con in shannonCons])))

transceivers = ["fixed-rate", "flex-rate", "shannon"]

snr = {"fixed-rate": [], "flex-rate": [], "shannon": []}


for transceiver in transceivers:
    allocated_capacity = []
    tot_cap = 0
    for m in range(1, 80):
        # traffic matrix
        net = Network("274328.json", 10, transceiver)

        traffic_m = np.random.randn(len(net.nodes) ** 2) * 100e9 * m
        traffic_m = np.full(len(net.nodes) ** 2, 100e9 * m)
        traffic_m[traffic_m < 0] = 0
        traffic_m = np.reshape(traffic_m, (len(net.nodes), len(net.nodes)))
        np.fill_diagonal(traffic_m, 0.0)  # exclude the diagonal part

        net.manageTrafficMatrix(traffic_m)

        best = None

        for line in net.lines.values():
            if best is None:
                best = line
            elif count_false(line.state) > count_false(best.state):
                best = line

        allocated_capacity.append(net.total_allocated_capacity())
        snr[transceiver].append(net.average_snr())

        tot_cap += net.total_allocated_capacity()

    plt.plot(allocated_capacity)
    plt.title(transceiver + " total allocated capacity")
    plt.show()
    print("total allocated capacity of " + transceiver + "is: " + str(tot_cap))
    tot_cap = 0

    plt.plot(snr[transceiver])
    plt.title(transceiver + "average snr")
    plt.show()
