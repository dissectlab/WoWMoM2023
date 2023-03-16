import math

# 8 Mbps
# 1881600
local_computation_time= [0.0102, 47.6049, 40.4949, 79.3638, 82.4224, 91.248, 131.8877, 169.7778, 296.7479, 337.7439, 345.364]
network_delay= [648.6486, 834.9912, 200.3565, 599.0856, 157.9804, 276.9765, 182.0593, 69.3065, 15.0938, 15.2577, 0]
remote_computation_time= [29.0507, 27.892, 28.0988, 22.7438, 22.3614, 18.9474, 15.3101, 12.5744, 4.4013, 1.071, 0]
total= [677.7095, 910.4881, 268.9502, 701.1932, 262.7642, 387.1719, 329.2571, 251.6587, 316.243, 354.0726, 0]
pow= [2272.1944, 2776.1967, 2746.8929, 2834.7545, 2809.7024, 2921.5, 2839.907, 2685.7625, 2769.5, 2830.619, 2804.061]

energy = []
for i in range(len(local_computation_time)):
    energy.append([local_computation_time[i] * pow[i]/math.pow(10, 6), 0.5 * network_delay[i]/1000])

s = "["
for i in range(len(energy)):
    if i != len(energy) -1:
        s += str(energy[i]) + ";"
    else:
        s += str(energy[i]) + "]"
print(s)

# 20 Mbps
# 1881600
local_computation_time= [0.0165, 37.8967, 37.1404, 74.0706, 78.6209, 96.9768, 130.9201, 167.0467, 271.115, 330.7657, 311.4781]
network_delay= [286.3999, 343.4938, 194.4293, 252.3966, 161.3659, 146.6603, 122.7495, 72.838, 9.7052, 6.9809, 0]
remote_computation_time= [31.3011, 28.0272, 26.4969, 22.7733, 21.9052, 18.924, 15.3169, 12.2512, 4.3896, 1.0545, 0]
total= [317.7175, 409.4177, 258.0666, 349.2405, 261.892, 262.5611, 268.9865, 252.1359, 285.2098, 338.8011, 0]
pow= [2307.1279, 2795.9891, 2709.0854, 2877.5568, 2728.5732, 2703.9268, 2740.2561, 2703.1125, 2704.575, 2829.2561, 2810.4634]

energy = []
for i in range(len(local_computation_time)):
    energy.append([local_computation_time[i] * pow[i]/math.pow(10, 6), 0.5 * network_delay[i]/1000])

s = "["
for i in range(len(energy)):
    if i != len(energy) -1:
        s += str(energy[i]) + ";"
    else:
        s += str(energy[i]) + "]"
print(s)

t = []
for i in range(len(local_computation_time)):
    t.append([local_computation_time[i], network_delay[i], remote_computation_time[i]])

s = "["
for i in range(len(local_computation_time)):
    if i != len(local_computation_time) -1:
        s += str(t[i]) + ";"
    else:
        s += str(t[i]) + "]"

print(s)