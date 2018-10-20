import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Input list for Convolutional code
K = 4           # constraint length
r = 2           # number of output bits per input bit
m = K-1         # memory size
# g = np.array([[1, 1, 1], [1, 1, 0]])
g = np.array([[1, 1, 0, 1], [1, 1, 1, 1]])
matlab_index = 1
rate_num = 3
rate_den = 4
puncturing_en = False
puncturing_matrix = np.array([[1, 1, 0], [1, 0, 1]])

data_length = 4
dout = np.zeros(data_length)

dout = np.array([1, 0, 1, 1], dtype=np.bool)
print(dout)
# shift_registers = np.zeros(K, dtype=np.bool)
shift_registers = np.array([1, 1, 0, 1])

# including zero padding due to zero termination
cc_out = np.zeros((r, data_length))

# convolutional coding
for idx in range(data_length):
    for mem in reversed(range(K)):
        shift_registers[mem] = shift_registers[mem-1]
    shift_registers[0] = dout[idx]
    for jdx in range(r):
        for kdx in range(K):
            if g[jdx][kdx] != 0:
                cc_out[jdx][idx] = np.logical_xor(cc_out[jdx][idx], shift_registers[kdx])

print(cc_out[0])
print(cc_out[1])

# puncturing
serialized_out = np.zeros(r * data_length)
cnt = 0
nrow_pmat, ncol_pmat = puncturing_matrix.shape
mark = 0
puncturing_per_output = nrow_pmat * ncol_pmat / r
for idx in range(data_length):
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
