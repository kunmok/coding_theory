import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Input list for Convolutional code
K = 3           # constraint length
r = 2           # number of output bits per input bit
m = K-1         # memory size
feedback = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.bool)
feedback_num = feedback[0]
feedback_den = feedback[1]
g = np.array([[1], feedback])
matlab_index = 1
rate_num = 3
rate_den = 4
puncturing_en = True
puncturing_matrix = np.array([[1, 1, 0], [1, 0, 1]])

data_length = 6
dout = np.zeros(data_length+(K-1))

# two different methods: put the data into the window
# or move the sliding window

# important question : how to create a state machine diagram using p0,p1,... ?
dout = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.bool)
print(dout)
shift_registers = np.zeros(K, dtype=np.bool)

# including zero padding due to zero termination
cc_out = np.zeros((r, data_length + (K-1)))

# convolutional coding
for idx in range(data_length + (K-1)):
    for mem in reversed(range(K)):
        shift_registers[mem] = shift_registers[mem-1]
    shift_registers[0] = dout[idx] + (shift_registers[1:]*feedback_den[1:]).sum() % 2
    for jdx in range(r):
        if len(g[jdx]) == 1:
            cc_out[jdx][idx] = dout[idx]
        else:
            for kdx in range(K):
                if feedback_num[kdx] != 0:
                    cc_out[jdx][idx] = np.logical_xor(cc_out[jdx][idx], shift_registers[kdx])

print(cc_out[0])
print(cc_out[1])

# ### print serialized output ###
serialized_out = np.zeros(r * (data_length + (K-1)))
cnt = 0
for idx in range(data_length + (K-1)):
    for jdx in range(r):
        serialized_out[cnt] = cc_out[jdx][idx]
        cnt = cnt + 1

print(serialized_out)

# ### Puncturing ###
serialized_out = np.zeros(r * (data_length + (K-1)))
cnt = 0
nrow_pmat, ncol_pmat = puncturing_matrix.shape
mark = 0
puncturing_per_output = nrow_pmat * ncol_pmat / r
for idx in range(data_length + (K-1)):
    for jdx in range(r):
        if puncturing_en:
            if puncturing_matrix[jdx][mark] > 0:
                serialized_out[cnt] = cc_out[jdx][idx]
                cnt = cnt + 1
        else:
            # serialized_out[idx*r + jdx] = cc_out[jdx][idx]
            serialized_out[cnt] = cc_out[jdx][idx]
            cnt = cnt + 1
    mark = int((mark + 1) % puncturing_per_output)

print(serialized_out)
