import math

import numpy as np


def cpu_transfer(c_allocation, total_number):
    s = []
    for i in c_allocation:
        s.append(i * total_number)
    #print(s)

    t = []
    d = []
    for i in s:
        t.append(math.floor(i))
        d.append(i - math.floor(i))

    #print(t, sum(t))
    #print(d)

    d = np.array(d)
    for index in d.argsort()[-(total_number - sum(t)):][::-1]:
        t[index] += 1
    print(t)
    cid = 0
    s = []
    for cpu in t:
        c_s = ""
        for i in range(cpu):
            if i < cpu-1:
                c_s += str(cid) + ","
            else:
                c_s += str(cid)
            cid += 1
        s.append(c_s)
    print(s)
    return s


#cpu_transfer([0.3, 0.2, 0.1, 0.3, 0.1], 16)

