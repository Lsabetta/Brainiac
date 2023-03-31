import pickle as pkl
import matplotlib.pyplot as plt
import glob
import torch
import sklearn
import sklearn.metrics
from opt import OPT
import numpy as np

OPT.DISTANCE_TYPEs = ["l2", "inverse_cosine"]#, "normalized_l2"]#["l2", "inverse_cosine"]
labels = ["l2", "CosDist"]#, "Normalized l2"]
#lens_labels = [len(l) for l in labels]
#labels = [l + " "*(max(lens_labels)-len(l)) for l in labels]
OPT.DATASET = 'CORE50'
OPT.MODEL = "openCLIP"
OPT.PROCESSING_FRAMES = 1
OPT.UPDATE_PROBABILITY = 1
colors_dots = ["#E37D43", '#FFB6C1']#, '#32212B']
colors_line = ['#98292B', '#C1E1C1']#, '#428362']
colors_fill = ["#26868E", "#77B342"]#, '#ADD8E6']

for i, d in enumerate(OPT.DISTANCE_TYPEs):
    OODs = [1] #OOD = Recall
    precision = [0] #precision
    distances_from_v = [1]
    color_dots = colors_dots[i]
    color_line = colors_line[i]
    color_fill = colors_fill[i]
    dir_path = f"/home/luigi/Work/Brainiac/paper/results/{OPT.DATASET}_{d}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}_{OPT.SHUFFLED_SCENARIOS}_p{int(OPT.UPDATE_PROBABILITY*100)}_sl{OPT.SELF_LEARNING}"
    print(dir_path)
    paths = glob.glob(dir_path + "/*.pkl")
    thresholds =  [float(".".join(p.split("_")[-1].split(".")[:-1])[1:]) for p in paths]
    thresholds.sort()
    path = dir_path + f"/matrix_t"
    matrices_paths = [path + f"{format(t, '.2f')}.pkl" for t in thresholds]
    print(thresholds)
    for j, p in enumerate(matrices_paths):
        with open(p, "rb") as f:
            m = pkl.load(f)
            precision.append(m.ood_confusion_matrix[0, 0]/(m.ood_confusion_matrix[0, 0]+m.ood_confusion_matrix[0, 1]))
            OODs.append(m.ood_confusion_matrix[0, 0]/(m.ood_confusion_matrix[0, 0]+m.ood_confusion_matrix[1, 0]))
            print(precision[-1], OODs[-1])
    thresholds = [0] + thresholds
    precision.append(1.)
    OODs.append(0.)
    
    AUC = sklearn.metrics.auc(precision, OODs)
   
    plt.plot(precision, OODs, color = color_line, lw = "2", label = f"{labels[i]}\nAUC = {round(AUC, 4)}, ACC = {round(m.accuracy(), 4)}")
    plt.fill_between(precision, OODs, color = color_fill, alpha = 0.5)
    for k, t in enumerate(thresholds):
        if t == 0:
            continue
        print(t, precision[k], OODs[k])
        distances_from_v.append(np.sqrt((1-OODs[k])**2 + (1-precision[k])**2))
    min_idx = np.argmin(distances_from_v)
    
    plt.plot(precision[min_idx], OODs[min_idx], color = color_dots, marker = "o", )
    plt.text(0.05+precision[min_idx], OODs[min_idx], f"t = {str(round(thresholds[min_idx], 2))}", fontsize = 12)
   

plt.legend(loc = 'lower left')
xmin = 0.
ymin = 0.
plt.plot(torch.arange(xmin, 1.1, 0.1).numpy()[::-1], torch.arange(xmin, 1.1, 0.1),color = color_line, ls = "--", lw = "3")

plt.xlim([xmin, 1.1])
plt.ylim([ymin, 1.1])
#plt.xscale("log")
#plt.yscale("log")
plt.grid()

plt.ylabel('OOD')
plt.xlabel('Precision')
#plt.show()

plt.savefig(f"{ '/'.join(dir_path.split('/')[:-1])}/OODs_vs_precision_{OPT.DATASET}.pdf")
