from network import *

network = Network("274328.json", 10)
network.draw()
nodes = list(network.nodes.keys())
cons = []
for i in range(0, 80):
    s = floor(random.uniform(0, len(nodes)))
    e = floor(random.uniform(0, len(nodes)))
    while e == s:
        e = floor(random.uniform(0, len(nodes)))
    cons.append(Connection(nodes[s], nodes[e], 1e-3))
network.stream(cons)
fig, [plot_latency, plot_snr] = plt.subplots(2)
plot_latency.plot(list(map(lambda x: x.latency, cons)))
plot_latency.set_title("Connection latency")
plot_snr.plot(list(map(lambda x: x.snr, cons)))
plot_snr.set_title("Connection GSNR")

plt.show()

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
fig, plot_speed = plt.subplots()
plot_speed.plot(db_a, speed_fixed_a, label="Fixed Speed")
plot_speed.plot(db_a, speed_flex_a, label="Flex Speed")
plot_speed.plot(db_a, speed_shannon_a, label="Shannon Speed")

plot_speed.legend()

plot_speed.set_xlabel("GSNR [dB]")
plot_speed.set_ylabel("Speed [Gb/s]")

plt.show()

plt.hist(list(map(lambda x: x.bitRate, cons)), bins=2)
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

plt.hist(list(map(lambda x: x.bitRate / 1e9, fixedCons)), bins=2)
plt.xlabel("bit Rate Gb/s")
plt.ylabel("Connections")
plt.title("Fixed-rate transceiver")
plt.show()

plt.hist(list(map(lambda x: x.bitRate / 1e9, flexCons)), bins=4)
plt.xlabel("bit Rate Gb/s")
plt.ylabel("Connections")
plt.title("Flex-rate transceiver")
plt.show()

plt.hist(list(map(lambda x: x.bitRate / 1e9, shannonCons)), bins=20)
plt.xlabel("bit Rate Gb/s")
plt.ylabel("Connections")
plt.title("shannon transceiver")
plt.show()

    # only maintain accepted connections
fixedCons = [con for con in fixedCons if con.bitRate > 0]
flexCons = [con for con in flexCons if con.bitRate > 0]
shannonCons = [con for con in shannonCons if con.bitRate > 0]

    # print average speeds
print("Average speed for fixed transceivers is: " + str(average([con.bitRate for con in fixedCons])))
print("Average speed for flex transceivers is: " + str(average([con.bitRate for con in flexCons])))
print("Average speed for shannon transceivers is: " + str(average([con.bitRate for con in shannonCons])))

print("Average snr for fixed transceivers is: " + str(calculate_average([con.snr for con in fixedCons])))
print("Average snr for flex transceivers is: " + str(calculate_average([con.snr for con in flexCons])))
print("Average snr for shannon transceivers is: " + str(calculate_average([con.snr for con in shannonCons])))

print("Average latency for fixed transceivers is: " + str(calculate_average([con.latency for con in fixedCons])))
print("Average latency for flex transceivers is: " + str(calculate_average([con.latency for con in flexCons])))
print("Average latency for shannon transceivers is: " + str(calculate_average([con.latency for con in shannonCons])))



transceivers = ["fixed-rate", "flex-rate", "shannon"]

snr_histories = {"fixed-rate": [], "flex-rate": [], "shannon": []}
snr_histories_after_recovery = {"fixed-rate": [], "flex-rate": [], "shannon": []}

for transceiver in transceivers:
    allocated_capacity_history = []
    allocated_capacity_after_recovery_history = []
    for m in range(1, 80):
         # begging the creation of the traffic matrix
        net = Network("274328.json", 10, transceiver)  # resetting the network

        Tm = np.random.randn(len(net.nodes) ** 2) * 100e9 * m
        Tm = np.full(len(net.nodes) ** 2, 100e9 * m)
        Tm[Tm < 0] = 0
        Tm = np.reshape(Tm, (len(net.nodes), len(net.nodes)))
        np.fill_diagonal(Tm, 0.0)
        #print(Tm)

        net.manageTrafficMatrix(Tm)
        #print(Tm)

        best = None

        for line in net.lines.values():
            if best is None:
                best = line
            elif (count_false(line.state) > count_false(best.state)):
                best = line

        allocated_capacity_history.append(net.total_allocated_capacity())
        snr_histories[transceiver].append(net.average_snr())

        net.strong_failure(best.label)

        print("Breaking line " + best.label)

        net.traffic_recovery()

        allocated_capacity_after_recovery_history.append(net.total_allocated_capacity())
        snr_histories_after_recovery[transceiver].append(net.average_snr())

    plt.plot(allocated_capacity_history, label="before recovery")
    plt.plot(allocated_capacity_after_recovery_history, label="after recovery")

    plt.legend()

    plt.title(transceiver + " total allocated capacity")

    plt.show()

    plt.plot(snr_histories[transceiver], label="snr before recovery")
    plt.plot(snr_histories_after_recovery[transceiver], label="snr after recovery")

    plt.legend()

    plt.title(transceiver + " average snr")
    plt.xlabel("m")
    plt.ylabel("average snr [dB]")

    plt.show()

