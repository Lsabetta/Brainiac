import pickle as pkl
import matplotlib.pyplot as plt
import glob
import torch

OODs = [1]
confusions = [1]
paths = glob.glob("results/thresholds/*")
thresholds =  [float(".".join(p.split("_")[-1].split(".")[:-1])[1:]) for p in paths]
for n in [3.0, 3.5, 4.0, 4.5, 5.0, 6.3]:
    thresholds.pop(thresholds.index(n))
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
        OODs.append(m.ood())
thresholds = [0] + thresholds
#confusions.append(0.)
#OODs.append(0.)

plt.plot(confusions, OODs, 'b')
plt.fill_between(confusions, OODs)
for i, t in enumerate(thresholds):
    if t == 0:
        continue
    plt.annotate(str(round(t, 2)), (confusions[i], OODs[i]), xytext=(-15, 5), textcoords = "offset points")
    print(t, confusions[i], OODs[i]) 
plt.legend(loc = 'lower right')

xmin = 0.005
#plt.plot(torch.arange(xmin, 1, 0.001), torch.arange(xmin, 1, 0.001).sort(descending=True)[0],'r--')
plt.plot(torch.arange(xmin, 1, 0.001), torch.arange(xmin, 1, 0.001),'r--')

plt.xlim([xmin, 1])
plt.ylim([xmin, 1])
plt.xscale("log")
#plt.yscale("log")
plt.grid()
plt.ylabel('OOD (%)')
plt.xlabel('Confusion (%)')
#plt.show()
plt.savefig("prova.pdf")