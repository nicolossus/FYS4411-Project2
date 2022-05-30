from pyvis.network import Network
net = Network()
for i in range(5):
    net.add_node(i, color="lightblue", label="Visible")
for i in range(5, 10, 1):
    net.add_node(i, color="pink", label="Hidden")

for i in range(10):
    for j in range(10):
        if (i==j):
            pass
        else:
            net.add_edge(i, j, physics=False)

net.show("visual.html")
