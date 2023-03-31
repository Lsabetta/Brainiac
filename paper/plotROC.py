import pickle as pkl
import matplotlib.pyplot as plt
import glob
import torch
import sklearn
import sklearn.metrics
from opt import OPT
import numpy as np

OPT.DISTANCE_TYPEs = ["normalized_l2"]#["l2", "inverse_cosine"]
labels = ["normalized l2"]#["l2", "CosDist"]
#lens_labels = [len(l) for l in labels]
#labels = [l + " "*(max(lens_labels)-len(l)) for l in labels] 
OPT.MODEL = "openCLIP"
OPT.PROCESSING_FRAMES = 1
OPT.UPDATE_PROBABILITY = 1
colors_dots = ["#E37D43", '#FFB6C1']
colors_line = ['#98292B', '#C1E1C1']
colors_fill = ["#26868E", "#ADD8E6"]

for i, d in enumerate(OPT.DISTANCE_TYPEs):
    OODs = [1]
    type1_ood_errors = [1]
    distances_from_v = [1]
    color_dots = colors_dots[i]
    color_line = colors_line[i]
    color_fill = colors_fill[i]
    dir_path = f"/home/leonardolabs/Documents/Brainiac/paper/results/{OPT.DATASET}_{d}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}_{OPT.SHUFFLED_SCENARIOS}_p{int(OPT.UPDATE_PROBABILITY*100)}_sl{OPT.SELF_LEARNING}"
    print(dir_path)
    paths = glob.glob(dir_path + "/*.pkl")
    thresholds =  [float(".".join(p.split("_")[-1].split(".")[:-1])[1:]) for p in paths]
    thresholds.sort()
    path = dir_path + f"/matrix_t"
    matrices_paths = [path + f"{format(t, '.2f')}.pkl" for t in thresholds]

    for j, p in enumerate(matrices_paths):
        with open(p, "rb") as f:
            m = pkl.load(f)
            type1_ood_errors.append(m.type1_ood_error())
            OODs.append(m.ood())
    thresholds = [0] + thresholds
    type1_ood_errors.append(0.)
    OODs.append(0.)
    
    AUC = sklearn.metrics.auc(type1_ood_errors, OODs)
   
    plt.plot(type1_ood_errors, OODs, color = color_line, lw = "2", label = f"{labels[i]}\nAUC = {round(AUC, 4)}, ACC = {round(m.accuracy(), 4)}")
    plt.fill_between(type1_ood_errors, OODs, color = color_fill, alpha = 0.8)
    for k, t in enumerate(thresholds):
        if t == 0:
            continue
        print(t, type1_ood_errors[k], OODs[k])
        distances_from_v.append(np.sqrt((0.05*(1-OODs[k]))**2 + (type1_ood_errors[k])**2))
    min_idx = np.argmin(distances_from_v)
    
    plt.plot(type1_ood_errors[min_idx], OODs[min_idx], color = color_dots, marker = "o", )
    plt.text(type1_ood_errors[min_idx], 0.6*OODs[min_idx], f"t = {str(round(thresholds[min_idx], 2))}", color = "w", fontsize = 12)
   

plt.legend(loc = 'lower left')
xmin = 0.001
ymin = 0.
plt.plot(torch.arange(xmin, 1, 0.001), torch.arange(xmin, 1, 0.001),color = color_line, ls = "--", lw = "3")

plt.xlim([xmin, 1])
plt.ylim([ymin, 1])
plt.xscale("log")
#plt.yscale("log")
plt.grid()

plt.ylabel('OOD')
plt.xlabel('Type 1 OOD err.')
#plt.show()

plt.savefig(f"/home/leonardolabs/Documents/Brainiac/paper/results/OODs_vs_type1_ood_error_{OPT.DATASET}.pdf")
