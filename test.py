import re
import math
import torch
from pprint import pprint
# print(torch.cuda.is_available())
# import dgl



# GAMMA0 = 0.1
# ALPHA0 = 0.5
# CL_EPOCH_NUM = 40
# SOFT_CL_WEIGHT_LS = [None, None, ]

# for l in range(2, 100):
#     SOFT_CL_WEIGHT_LS.append([None, ])
#     for i in range(1, CL_EPOCH_NUM + 1):
#         SOFT_CL_WEIGHT_LS[-1].append([])
#         gamma_i = GAMMA0 + i / CL_EPOCH_NUM * (1 - GAMMA0)
#         for t in range(1, l + 1):
#             alpha_t_l = ALPHA0 * (t - 1) / (l - 1)
#             weight = gamma_i ** alpha_t_l
#             SOFT_CL_WEIGHT_LS[-1][-1].append(weight)

# l = 20
# for i in range(1, CL_EPOCH_NUM + 1):
#     print(f'epoch {i}\n   ', end='')
#     for t in range(l):
#         print(round(SOFT_CL_WEIGHT_LS[l][i][t], 3), end=' ')
#     print('\n')

GAMMA0 = 0.1
ALPHA0 = 0.5
CL_EPOCH_NUM = 40
SOFT_CL_WEIGHT_LS = [None, ]

for i in range(1, CL_EPOCH_NUM + 1):
    SOFT_CL_WEIGHT_LS.append([None, None, ])
    gamma_i = GAMMA0 + i / CL_EPOCH_NUM * (1 - GAMMA0)
    for l in range(2, 100):
        SOFT_CL_WEIGHT_LS[-1].append([])
        for t in range(1, l + 1):
            alpha_t_l = ALPHA0 * (t - 1) / (l - 1)
            weight = gamma_i ** (alpha_t_l * 5)
            SOFT_CL_WEIGHT_LS[-1][-1].append(weight)

l = 20
for i in range(1, CL_EPOCH_NUM + 1):
    if i in [1, 20, 39, 40]:
        print(f'epoch {i}:\n   ', end='')
        for t in range(l):
            print(round(SOFT_CL_WEIGHT_LS[i][l][t], 3), end=' ')
        print('\n')