import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

#!!!!!!!!!!TARGET::::::-----> 计算（拟合）概率密度函数和概率分布函数 for 连续性随机变量

x = np.append(np.random.randint(0, 100, 1000), np.random.randint(0,200, 500))
x.sort()

print("Mean: %.2f" % np.mean(x))
print("Median: %.2f" % np.median(x))
print("Mode: %.2f" % stats.mode(x))
print("Ex diff: %.2f" % np.ptp(x))
print("Var: %.2f" % np.var(x))
print("Std: %.2f" % np.std(x))
print("Var coff: %.2f" % np.mean(x)/np.std(x))

print("Deviation degree(Z score)" % (x[0] - np.mean(x))/np.std(x))  # see first val if val > 3, seems like the error val
#list(map(zScore, x, [np.mean(x)] * x_size, x_size * [np.std(x)]))



def zScore(val, mean, std):
    return (val - mean)/std

# plt.plot(x,y)
# plt.title('Normal: $\mu$=%.1f, $\sigma^2$=%.1f' % (mu, sigma))
# plt.xlabel('x')
# plt.ylabel('Proper density')
# plt.show()

x = np.random.normal(0, 1, size=30)
x_1 = np.linspace(.5, 1, 20)
x_2 = np.linspace(-1, -.5, 20)
x = np.append(x, x_1)
x = np.append(x, x_2)

bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)
#x.min(), x.max()
support = np.linspace(-4, 4, 2000)

kernels = []
for x_i in x:

    kernel = stats.norm.pdf(support, x_i, bandwidth)
    kernels.append(kernel)
    # plt.plot(support, kernel, color="r")

# sns.rugplot(x, color=".2", linewidth=3);

from scipy.integrate import trapz
density = np.sum(kernels, axis=0)
density /= trapz(density, support)
plt.plot(support, density);
# area_almost = np.trapz(density, x=support)

#point is the val of needed point
# plt.fill_between(support[index], 0, density[index], alpha=.5)
index = np.where(np.logical_and(support > point - .5, support <= point + .5))
area_almost = np.trapz(density[index], x=support[index])



