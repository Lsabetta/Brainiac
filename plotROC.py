import pickle as pkl
import matplotlib.pyplot as plt
import glob
import torch

OODs = [0.]
confusions = [1.]
paths = glob.glob("results/thresholds/*")
thresholds = [float(".".join(p.split("_")[-1].split(".")[:-1])[1:]) for p in paths]
thresholds.sort()
#thresholds = torch.arange(5, 8, 0.5).numpy()
path = "results/thresholds/matrix_t"
matrices_paths = [path + f"{t}.pkl" for t in thresholds]
for i, p in enumerate(matrices_paths):
    if thresholds[0]<5:
        thresholds.pop(0)
        continue
    with open(p, "rb") as f:
        m = pkl.load(f)
        #thresholds.append(float("".join(p.split("_")[-1].split(".")[:-1])[1:]))
        confusions.append(m.confusion())
        OODs.append(1-m.ood())
confusions.append(0.)
OODs.append(1.)

plt.plot(confusions, OODs, 'b')
#plt.fill_between(confusions, OODs)
for i, t in enumerate(thresholds):
    plt.annotate(str(round(t, 2)), (confusions[i], OODs[i]), xytext=(-15, 5), textcoords = "offset points") 
plt.legend(loc = 'lower right')
xmin = 0.005
plt.plot(torch.arange(1, 1, 0.001), torch.arange(xmin, 1, 0.001),'r--')
plt.xlim([xmin, 1])
plt.ylim([xmin, 1])
plt.xscale("log")
#plt.yscale("log")
plt.grid()
plt.ylabel('OOD')
plt.xlabel('Confusion')
plt.show()
plt.savefig("prova.pdf")