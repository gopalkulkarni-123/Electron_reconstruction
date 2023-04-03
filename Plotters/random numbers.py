import  numpy as np

a = [0.2, 0.3, 0.1, 0.4]

n = 4
data = []
size = 100
for i in range(100):
    flow = np.random.choice(range(5), size = n, p = a)
    data.append(flow)

print(data)