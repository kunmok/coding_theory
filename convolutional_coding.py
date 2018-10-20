import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Input list for Convolutional code
K = 3           # constraint length
r = 2           # output rate
m = K-1         # memory size
g = np.array([[1,1,1],[1,1,0]])
matlab_index = 1
puncturing_codeword = 3
puncturing_p1 = 2 - matlab_index
puncturing_p2 = 3 - matlab_index

data_length = 6
dout = np.zeros(data_length+(K-1))

# start_state = np.uint32(0xACE1)
start_state = np.uint32(0xF7FFFFFF)
lfsr = np.uint32(start_state)
for x in range(data_length):
    newbit = np.uint32(((lfsr >> 30)^(lfsr >> 27)) & 0x1)
    lfsr = (lfsr << 1) | newbit
    lfsr = lfsr & 0x7FFFFFFF
    if newbit > 0.5:
        dout[x] = 1
    else:
        dout[x] = 0

# plt.plot(dout)
# plt.show()

# two different methods: put the data into the window
# or move the sliding window

# important question : how to create a state machine diagram using p0,p1,... ?
dout = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.bool)
print(dout)
sliding_window = np.zeros(K, dtype=np.bool)

cc_out = np.zeros((r, data_length + (K-1)))

for idx in range(data_length + (K-1)):
    for mem in reversed(range(K)):
        sliding_window[mem] = sliding_window[mem-1]
    sliding_window[0] = dout[idx]
    for jdx in range(r):
        for kdx in range(K):
            if g[jdx][kdx] != 0:
                cc_out[jdx][idx] = np.logical_xor(cc_out[jdx][idx], sliding_window[kdx])

print(cc_out[0])
print(cc_out[1])

# puncturing
serialized_out = np.zeros(r * (data_length + (K-1)))
for idx in range(data_length + (K-1)):
    for jdx in range(r):
        serialized_out[idx*r + jdx] = cc_out[jdx][idx]

print(serialized_out)
# implement shift register


