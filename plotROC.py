import pickle as pkl
import matplotlib.pyplot as plt
import glob
import torch
import sklearn
import sklearn.metrics
from opt import OPT
OODs = [1]
confusions = [1]
OPT.DISTANCE_TYPE = "normalized_l2"
OPT.MODEL = "CLIP"
OPT.PROCESSING_FRAMES = 30
paths = glob.glob(f"results/{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}/*")
thresholds =  [float(".".join(p.split("_")[-1].split(".")[:-1])[1:]) for p in paths]
thresholds.sort()
path = f"results/{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}/matrix_t"

matrices_paths = [path + f"{format(t, '.2f')}.pkl" for t in thresholds]
#

for i, p in enumerate(matrices_paths):
    with open(p, "rb") as f:
        m = pkl.load(f)
        confusions.append(m.confusion())
        OODs.append(m.ood())
thresholds = [0] + thresholds
confusions.append(0.)
OODs.append(0.)

plt.plot(confusions, OODs, 'b')
plt.fill_between(confusions, OODs)
for i, t in enumerate(thresholds):
    if t == 0:
        continue
    if i%1 == 0:
        plt.plot(confusions[i], OODs[i], "ro")
        plt.text(confusions[i], 1.1*OODs[i], f"t = {str(round(t, 2))}")
        #plt.annotate(f"t = {str(round(t, 2))}", (confusions[i], OODs[i]), xytext=(-15, 5), textcoords = "offset points", marker=".")
    print(t, confusions[i], OODs[i]) 
plt.legend(loc = 'lower right')
AUC = sklearn.metrics.auc(confusions, OODs)
xmin = 0.001
ymin = 0.001
plt.plot(torch.arange(xmin, 1, 0.001), torch.arange(xmin, 1, 0.001),'r--')

plt.xlim([xmin, 1])
plt.ylim([ymin, 1])
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.text(0.3, 0.2, f"AUC = {round(AUC, 4)}\nACC = {round(m.accuracy(), 4)}", usetex=True)

plt.ylabel('OOD (%)')
plt.xlabel('Confusion (%)')
#plt.show()
plt.savefig(f"results/{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}_OODs_vs_confusion.pdf")