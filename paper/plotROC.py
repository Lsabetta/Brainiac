import pickle as pkl
import matplotlib.pyplot as plt
import glob
import torch
import sklearn
import sklearn.metrics
from opt import OPT
import matplotlib as mpl

OODs = [1]
type1_ood_errors = [1]
OPT.DISTANCE_TYPE = "inverse_cosine"
OPT.MODEL = "openCLIP"
OPT.PROCESSING_FRAMES = 1

dir_path = f"results/{OPT.DATASET}_{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}_{OPT.SHUFFLED_SCENARIOS}_p{int(OPT.UPDATE_PROBABILITY*100)}"
paths = glob.glob(dir_path + "/*.pkl")
thresholds =  [float(".".join(p.split("_")[-1].split(".")[:-1])[1:]) for p in paths]
thresholds.sort()
path = dir_path + f"/matrix_t"

matrices_paths = [path + f"{format(t, '.2f')}.pkl" for t in thresholds]
#

for i, p in enumerate(matrices_paths):
    with open(p, "rb") as f:
        m = pkl.load(f)
        type1_ood_errors.append(m.type1_ood_error())
        OODs.append(m.ood())
thresholds = [0] + thresholds
type1_ood_errors.append(0.)
OODs.append(0.)

plt.plot(type1_ood_errors, OODs, 'b')
plt.fill_between(type1_ood_errors, OODs)
for i, t in enumerate(thresholds):
    if t == 0:
        continue
    if i%1 == 0:
        plt.plot(type1_ood_errors[i], OODs[i], "ro")
        plt.text(type1_ood_errors[i], 1.1*OODs[i], f"t = {str(round(t, 2))}")
        #plt.annotate(f"t = {str(round(t, 2))}", (type1_ood_errors[i], OODs[i]), xytext=(-15, 5), textcoords = "offset points", marker=".")
    print(t, type1_ood_errors[i], OODs[i]) 
plt.legend(loc = 'lower right')
AUC = sklearn.metrics.auc(type1_ood_errors, OODs)
xmin = 0.001
ymin = 0.001
plt.plot(torch.arange(xmin, 1, 0.001), torch.arange(xmin, 1, 0.001),'r--')

plt.xlim([xmin, 1])
plt.ylim([ymin, 1])
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.text(0.2, 0.1, f"AUC = {round(AUC, 4)}\nACC = {round(m.accuracy(), 4)}")

plt.ylabel('OOD')
plt.xlabel('Type 1 OOD err.')
#plt.show()

plt.savefig(f"{dir_path}/OODs_vs_type1_ood_error.pdf")