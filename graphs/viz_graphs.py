import numpy as np
import matplotlib.pyplot as plt

f_inlier_th = open('inlier_threshold.txt', "r")
inlier_th_lines = f_inlier_th.readlines()
f_inlier_th.close()

f_distance_th = open('distance_threshold.txt', "r")
distance_th_lines = f_distance_th.readlines()
f_distance_th.close()

fig_size = (7, 8)

our_sens_inlier_perc0 = [0.9884, 0.9771, 0.9639, 0.9552, 0.9496, 0.9390, 0.9312, 0.9237, 0.9127, 0.9040, 0.8967, 0.8836, 0.8762, 0.8626, 0.8472, 0.8345, 0.8241, 0.8068, 0.7904, 0.7773, 0.7609]
our_sens_distance_perc0 = [0.0000, 0.0179, 0.3099, 0.6007, 0.7782, 0.8601, 0.9033, 0.9279, 0.9427, 0.9538, 0.9575, 0.9593, 0.9624, 0.9649, 0.9667, 0.9698, 0.9723, 0.9735, 0.9735, 0.9754, 0.9772]

our_sens_inlier_perc5 = [0.9884, 0.9769, 0.9631, 0.9562, 0.9500, 0.9388, 0.9333, 0.9241, 0.9173, 0.9075, 0.8999, 0.8864, 0.8787, 0.8694, 0.8556, 0.8474, 0.8316, 0.8154, 0.7985, 0.7840, 0.7698]
our_sens_distance_perc5 = [0.0000, 0.0197, 0.3210, 0.6168, 0.7868, 0.8663, 0.9082, 0.9279, 0.9452, 0.9526, 0.9581, 0.9612, 0.9624, 0.9643, 0.9680, 0.9710, 0.9723, 0.9735, 0.9735, 0.9766, 0.9778]

colors = np.asarray([[128, 128, 128],
                     [255, 178, 102],
                     [255, 153, 255],
                     [204, 102, 153],
                     [153, 102, 204],
                     [153, 255, 255],
                     [153, 204, 255],
                     [153, 153, 255],
                     [153, 255, 153],
                     [255, 102, 102],
                     [0, 0, 255],
                     [0, 0, 255]])/255

linestyles = ['-','-','-','-','-','-','-','-','-','-',':','-']

# inlier graph
inlier_ths = np.asarray(inlier_th_lines[0].split(' ')[1:], dtype=float)
methods = []
scores = np.empty((len(inlier_th_lines[1:]), len(inlier_ths)))
for l, line in enumerate(inlier_th_lines[1:]):
    methods.append(line.split(' ')[0])
    scores[l] = np.asarray(line.split(' ')[1:], dtype=float) / 100

methods.append('DIP (' + r'$\mathsf{p}_\rho=0$)')
methods.append('DIP (' + r'$\mathsf{p}_\rho=5$)')

scores = np.vstack((scores, our_sens_inlier_perc0, our_sens_inlier_perc5))

fig = plt.figure(1, figsize=fig_size)
for i, _ in enumerate(scores):
    plt.plot(inlier_ths, scores[i], color=colors[i], label=methods[i], linestyle=linestyles[i], linewidth=4)

plt.xlim(.01, .2)
plt.ylim(0, 1)

plt.xlabel('inlier ratio threshold', fontsize=14)
plt.ylabel('feature-match recall', fontsize=14)

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['.00', '.20', '.40', '.60', '.80', '1.0'])
plt.xticks([0.025, 0.050, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2], ['.03', '.05', '.08', '.10', '.13', '.15', '.18', '.20'])

plt.gca().xaxis.set_tick_params(labelsize=12)
plt.gca().yaxis.set_tick_params(labelsize=12)

plt.grid()


# distance graph
distance_ths = np.asarray(distance_th_lines[0].split(' ')[1:], dtype=float)
methods = []
scores = np.empty((len(distance_th_lines[1:]), len(distance_ths)))
for l, line in enumerate(distance_th_lines[1:]):
    methods.append(line.split(' ')[0])
    scores[l] = np.asarray(line.split(' ')[1:], dtype=float) / 100

methods.append('DIP (' + r'$\mathsf{p}_\rho=0$)')
methods.append('DIP (' + r'$\mathsf{p}_\rho=5$)')

scores = np.vstack((scores, our_sens_distance_perc0, our_sens_distance_perc5))

fig = plt.figure(2, figsize=fig_size)
for i, _ in enumerate(scores):
    plt.plot(distance_ths, scores[i], color=colors[i], label=methods[i], linestyle=linestyles[i], linewidth=4)

plt.xlim(0, .2)
plt.ylim(0, 1)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=6, loc='upper left', fancybox=True, fontsize=14)

plt.xlabel('inlier distance threshold [m]', fontsize=14)
plt.ylabel('feature-match recall', fontsize=14)

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['.00', '.20', '.40', '.60', '.80', '1.0'])
plt.xticks([0, 0.025, 0.050, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2], ['.00', '.03', '.05', '.08', '.10', '.13', '.15', '.18', '.20'])

plt.gca().xaxis.set_tick_params(labelsize=12)
plt.gca().yaxis.set_tick_params(labelsize=12)

plt.grid()


plt.show()