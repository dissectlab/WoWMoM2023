import math
import random

import numpy as np

"""
if dnn_type == "AlexNet":
            self.output_data_size = [0.5744, 0.7387, 0.1782, 0.5341, 0.124, 0.2477, 0.1652, 0.0353, 0.0158, 0.0158, 0.004]

            self.layer_complexity = [0, 0.0106, 0.0066, 0.0252, 0.005, 0.0157, 0.02, 0.0176, 0.0502, 0.0226, 0.0059]
            print(sum(self.layer_complexity))
            self.tx2_layer_complexity = [0, 0.0385, 0.0105, 0.0331, 0.0071, 0.0205, 0.0273, 0.0291, 0.1664, 0.0896, 0.0223]

        elif dnn_type == "VGG":
            self.output_data_size = [0.5744, 12.2502, 12.2502, 3.0627, 6.1252, 6.1252, 1.5314, 3.0627, 3.0627, 0.7658, 1.5314, 1.5314,
                    0.383, 0.383, 0.383, 0.0959, 0.0158, 0.0158, 0.004]

            self.layer_complexity = [0, 0.0372, 0.2332, 0.0477, 0.1128, 0.2098, 0.0252, 0.1064, 0.2029, 0.2166, 0.1052, 0.2056, 0.213, 0.0611, 0.0604, 0.0642, 0.162, 0.0289, 0.0073]
            self.tx2_layer_complexity = [0, 0.0773, 0.3426, 0.091, 0.1484, 0.257, 0.0515, 0.1144, 0.2028, 0.2391, 0.1025, 0.1844, 0.1935, 0.0552, 0.0592,
                0.0676, 0.442, 0.0856, 0.0213]
        elif dnn_type == "ResNet":
            self.layer_complexity = [0, 0.0127, 0.009, 0.0152, 0.0116, 0.0038, 0.0111, 0.0034, 0.0111, 0.0036, 0.0111, 0.0032, 0.0067, 0.0028, 0.0108, 0.0026, 0.0107, 0.0028, 0.0108, 0.0026, 0.0068, 0.0031, 0.0114, 0.0028, 0.0112, 0.0031, 0.0112, 0.0029, 0.009, 0.0041, 0.0162, 0.004, 0.0154, 0.0042, 0.0154, 0.0041, 0.0004, 0.001]
            self.output_data_size = [0.5744, 3.0627, 3.0627, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658,
                    0.383, 0.383, 0.383, 0.383, 0.383, 0.383, 0.383, 0.383, 0.1916, 0.1916, 0.1916, 0.1916, 0.1916,
                    0.1916, 0.1916, 0.1916, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0021,
                    0.004]
            self.tx2_layer_complexity = [0, 0.0469, 0.0318, 0.0403, 0.0233, 0.0156, 0.0238, 0.0105, 0.0236, 0.0157, 0.024, 0.0102, 0.0177, 0.0149, 0.0194, 0.0113, 0.0194, 0.0147, 0.0199, 0.0112, 0.0134, 0.0179, 0.0164, 0.0151, 0.0163, 0.0179, 0.0166, 0.015, 0.0154, 0.0243, 0.0253, 0.0241, 0.0276, 0.0248, 0.0249, 0.0241, 0.001, 0.0027]

"""


paras_server_complexity = {
    "RestNet": [0, 0.0127, 0.009, 0.0152, 0.0116, 0.0038, 0.0111, 0.0034, 0.0111, 0.0036, 0.0111, 0.0032, 0.0067, 0.0028, 0.0108, 0.0026, 0.0107, 0.0028, 0.0108, 0.0026, 0.0068, 0.0031, 0.0114, 0.0028, 0.0112, 0.0031, 0.0112, 0.0029, 0.009, 0.0041, 0.0162, 0.004, 0.0154, 0.0042, 0.0154, 0.0041, 0.0004, 0.001],
    "VGG": [0, 0.0372, 0.2332, 0.0477, 0.1128, 0.2098, 0.0252, 0.1064, 0.2029, 0.2166, 0.1052, 0.2056, 0.213, 0.0611, 0.0604, 0.0642, 0.162, 0.0289, 0.0073],
    "AlexNet": [0, 0.0106, 0.0066, 0.0252, 0.005, 0.0157, 0.02, 0.0176, 0.0502, 0.0226, 0.0059]
}

paras_local_complexity = {
    "RestNet": [0, 0.0381, 0.0344, 0.0422, 0.0272, 0.0164, 0.027, 0.0111, 0.0265, 0.0164, 0.0273, 0.011, 0.0171, 0.0156, 0.0225, 0.012, 0.0228, 0.0156, 0.0222, 0.0119, 0.0128, 0.0188, 0.0184, 0.0159, 0.0182, 0.0187, 0.0185, 0.0158, 0.0155, 0.0255, 0.0268, 0.025, 0.0275, 0.0257, 0.0267, 0.025, 0.0011, 0.0032],
    "VGG": [0, 0.0773, 0.3426, 0.091, 0.1484, 0.257, 0.0515, 0.1144, 0.2028, 0.2391, 0.1025, 0.1844, 0.1935, 0.0552, 0.0592, 0.0676, 0.442, 0.0856, 0.0213],
    "AlexNet": [0, 0.0385, 0.0105, 0.0331, 0.0071, 0.0205, 0.0273, 0.0291, 0.1664, 0.0896, 0.0223]
}

paras_dnn_data = {
    "RestNet": [0.5744, 3.0627, 3.0627, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.7658, 0.383, 0.383, 0.383, 0.383, 0.383, 0.383, 0.383, 0.383, 0.1916, 0.1916, 0.1916, 0.1916, 0.1916, 0.1916, 0.1916, 0.1916, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0959, 0.0021, 0.004],
    "VGG": [0.5744, 12.2502, 12.2502, 3.0627, 6.1252, 6.1252, 1.5314, 3.0627, 3.0627, 3.0627, 0.7658, 1.5314, 1.5314, 1.5314, 0.383, 0.383, 0.383, 0.383, 0.0959, 0.0959, 0.0158, 0.0158, 0.004],
    "AlexNet": [0.5744, 0.7387, 0.1782, 0.5341, 0.124, 0.2477, 0.1652, 0.0353, 0.0158, 0.0158, 0.004]
}

allow_feq = [345600, 499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200]
MAX_Feq = 2035200


def pick_feq(feq):
    for i in range(len(allow_feq)):
        if allow_feq[i] < feq * math.pow(10, 6):
            continue
        else:
            return allow_feq[i]/math.pow(10, 6)
    return allow_feq[-1]/math.pow(10, 6)


def tx2_computation_pow(feq, pow_k=1279.8434):
    return pow_k * (feq ** 2) / 1000.


def do_cpu_tx2(number_of_tx2, q, v, cut_point, pow_k=1279.8434):
    feq = []
    for tx2_id in range(number_of_tx2):
        #if cut_point[tx2_id] == 0:
        #    feq.append(pick_feq(0))
        #else:
        new_feq = math.pow(q[tx2_id] / (v * pow_k), 1. / 2)
        feq.append(pick_feq(min(new_feq, MAX_Feq)))
    return feq


def do_cpu_allocation(number_of_tx2, q, cut_point, types, server_feq):
    total_weights = 0
    X_edges = []
    c_allocation = []
    for tx2_id in range(number_of_tx2):
        complexity = 0
        for i in range(cut_point[tx2_id], len(paras_server_complexity[types[tx2_id]])):
            complexity += paras_server_complexity[types[tx2_id]][i]
        #print("xx", cut_point[tx2_id], complexity)
        t_ = q[tx2_id] * complexity
        X_edges.append(t_)
        total_weights += math.sqrt(X_edges[tx2_id])
    #print("################")
    c_allocation = [0 for i in range(number_of_tx2)]
    for tx2_id in range(number_of_tx2):
        if total_weights > 0:
            c_allocation[tx2_id] = server_feq * math.sqrt(X_edges[tx2_id]) / total_weights
        else:
            c_allocation[tx2_id] = 0
    #print("c:", round(total_weights, 4), c_allocation, X_edges)
    return c_allocation


def do_b_allocation(number_of_tx2, v, q, cut_point, types, tx2_tx_pow, server_bandwidth):
    total_weights = 0
    d = []
    b_allocation = [0 for i in range(number_of_tx2)]
    for tx2_id in range(number_of_tx2):
        d.append(paras_dnn_data[types[tx2_id]][cut_point[tx2_id]])
        total_weights += math.sqrt(d[-1] * (q[tx2_id] + v * tx2_tx_pow[tx2_id]))

    for tx2_id in range(number_of_tx2):
        if total_weights > 0:
            b_allocation[tx2_id] = server_bandwidth * math.sqrt(
                d[tx2_id] * (q[tx2_id] + v * tx2_tx_pow[tx2_id])) / total_weights
        else:
            b_allocation[tx2_id] = 0
    return b_allocation


def cal_network_delay(number_of_tx2, b_allocation, cut_point, types):
    delay = []
    for tx2_id in range(number_of_tx2):
        d = paras_dnn_data[types[tx2_id]][cut_point[tx2_id]]
        b = b_allocation[tx2_id]
        if b != 0:
            delay.append(round(d/b, 4) * 1000)
    return delay


def measure(number_of_tx2, tx2_feq, q, cut_point, types, tx2_tx_pow, c_allocation, b_allocation):
    local_latency = []
    remote_latency = []
    network_delay = []
    energy = []
    network_e = []
    total_latency = []
    for tx2_id in range(number_of_tx2):
        # local
        local_complexity = 0
        for i in range(cut_point[tx2_id]):
            local_complexity += paras_local_complexity[types[tx2_id]][i] * 1.3
        if local_complexity > 0 and tx2_feq[tx2_id] > 0:
            local_latency.append(1000 * local_complexity / tx2_feq[tx2_id])
        else:
            local_latency.append(0)
        # remote
        remote_complexity = np.sum(paras_server_complexity[types[tx2_id]][cut_point[tx2_id]:])
        if remote_complexity > 0 and c_allocation[tx2_id] > 0:
            remote_latency.append(1000 * remote_complexity / (c_allocation[tx2_id] * 3.2))
        else:
            remote_latency.append(0)
        # network
        if b_allocation[tx2_id] > 0:
            network_delay.append(1000 * paras_dnn_data[types[tx2_id]][cut_point[tx2_id]]/b_allocation[tx2_id])
        else:
            network_delay.append(0)
        #print(0.001 * local_latency[tx2_id] * tx2_computation_pow(tx2_feq[tx2_id]))
        network_e.append(network_delay[tx2_id] * tx2_tx_pow[tx2_id] * 0.001)
        energy.append(network_delay[tx2_id] * tx2_tx_pow[tx2_id] * 0.001 + 0.001 * local_latency[tx2_id] * tx2_computation_pow(tx2_feq[tx2_id]))
        total_latency.append(local_latency[tx2_id] + remote_latency[tx2_id] + network_delay[tx2_id])

    return total_latency, network_delay, energy, network_e


def select_cut_point(number_of_tx2, v, q, cut_point, types, tx2_tx_pow, tx2_feq, server_feq, server_bandwidth, deadline):
    tx2_id = 0 #random.randint(0, number_of_tx2 - 1)
    opt_cut_point = None
    min_obj = None
    obj_list = []
    latency_ = []
    t_latency_ = []
    e_ = []
    ne_ = []
    for cut in range(len(paras_server_complexity[types[tx2_id]])):
        cut_point[tx2_id] = cut
        c_allocation = do_cpu_allocation(number_of_tx2, q, cut_point, types, server_feq)
        b_allocation = do_b_allocation(number_of_tx2, v, q, cut_point, types, tx2_tx_pow, server_bandwidth)
        total_latency, network_delay, energy, network_e = measure(number_of_tx2, tx2_feq, q, cut_point, types, tx2_tx_pow, c_allocation, b_allocation)
        latency_.append(round(network_delay[tx2_id], 4))
        t_latency_.append(round(total_latency[tx2_id], 1))
        e_.append(round(energy[tx2_id], 4))
        ne_.append(round(network_e[tx2_id], 4))
        obj = v * np.average(energy)
        q_s = 0
        for i in range(number_of_tx2):
            q_s += q[i] * (total_latency[i] - deadline[i])/1000.
        obj += q_s / number_of_tx2
        obj_list.append(round(obj, 4))

        if opt_cut_point is None or min_obj > obj:
            min_obj = obj
            opt_cut_point = cut

    cut_point[tx2_id] = opt_cut_point
    print("#########################")
    print(f"select {cut_point},feq={tx2_feq[tx2_id]}")
    print("n_latency", latency_)
    print("t_latency", t_latency_)
    print("obj_list", obj_list)
    print("e_", e_)
    print("ne_", ne_)
    print("#########################")
    return cut_point


def cpu_transfer(c_allocation, total_number):
    s = c_allocation
    #print(s)

    t = []
    d = []
    for i in s:
        t.append(math.floor(i))
        d.append(i - math.floor(i))

    d = np.array(d)
    for index in d.argsort()[-(total_number - sum(t)):][::-1]:
        t[index] += 1
    #print(t)
    cid = 0
    s = []
    for cpu in t:
        c_s = ""
        for i in range(cpu):
            if i < cpu - 1:
                c_s += str(cid) + ","
            else:
                c_s += str(cid)
            cid += 1
        s.append(c_s)
    #print(s)
    return s