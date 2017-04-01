import numpy as np, sys
sys.path.insert(0, "/home/rakshit/Desktop/Academic/ms_codefundo/News_Recommender/policy/")
sys.path.insert(0, "/home/rakshit/Desktop/Academic/ms_codefundo/News_Recommender/environment/")
import utilities as utils
from glmUCB import GLMUCB
from adversary import Adversary
import matplotlib.pyplot as plt

X, Y, lb = utils.get_data()
adv = Adversary(Y, X, model = "forest cover", log_bias = lb)
model = GLMUCB(X)
print "DATA READY"

y_plot = []
# colors = ['red', 'green', 'blue', 'black']
colors = ['red', 'green', 'blue', 'black', 'brown', 'pink', 'orange', 'violet', 'cyan', 'yellow']
labels = []
plt.subplot(211)
lines = [0 for i in range(1, 11)]

i = -1
for ro in range(1, 11, 10):
	i += 1
	ro = float(ro) / 1000.0
	yp = []
	dist_diff = []
	regret = 0.0
	avg_regret = 0.0

	for t in range(1000):
		contexts = X
		ch = model.predict_arm(contexts)
		reward, reg = adv.get_adversary_reward(ch)
		model.pull_arm(ch, reward, contexts)

		regret += reg
		if t % 100 == 0:
			print regret, t/100
		avg_regret = regret / (t + 1)
		yp.append(regret)

	print avg_regret
	# print dist_diff[-1]
	labels.append("ro = " + str(ro))
	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
	y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
	# y_plot.append(dist_diff)
plt.ylabel("Cumulative regret")
plt.xlabel("time steps")

plt.subplot(212)
i = -1
for yp in y_plot:
	i += 1
	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
plt.ylabel("Average regret")
plt.xlabel("time steps")
plt.legend()
plt.show()
