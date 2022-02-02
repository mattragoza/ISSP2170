import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set_palette('pastel')
np.random.seed(1)

# figure settings
fig_w = 6.5
fig_h = 2

# load data set
data = pd.read_csv('gaussian1.txt')
data.columns = [c.strip() for c in data.columns]
print(data.describe())
x = data.value

# unbiased sample statistics
n = len(x)
x_bar = sum(x) / n
s2  = sum((x - x_bar)**2) / (n - 1)
s = np.sqrt(s2)
print(f'sample mean = {x_bar:.4f}')
print(f'sample var  = {s2:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
colors = sns.color_palette()
x_min =  -3
x_max = 10

# plot histogram
ax = axes[0]
bins = np.linspace(0, 10, 11)
sns.histplot(x, bins=bins, ax=ax)
ax.set_ylabel('Count')
ax.set_title('Histogram')

# plot MLE gaussian
ax = axes[1]
x = np.linspace(x_min, x_max, 1000)
y = scipy.stats.norm.pdf(x, x_bar, s)
ax.fill_between(x, 0, y, alpha=0.25)
ax.plot(x, y)
p_x_bar = scipy.stats.norm.pdf(x_bar, x_bar, s)
ax.vlines(x_bar, 0, p_x_bar)
ax.set_ylabel('$P(X)$')
ax.set_title(r'$N(x|\bar{x},s)$')

for ax in axes:
	ax.set_xlim(x_min, x_max)
	ax.set_xlabel('$X$')
	ax.set_axisbelow(True) # grid lines behind data
	ax.grid(True, linestyle=':', color='lightgray')
	ax.set_ylim(0, ax.get_ylim()[1])
sns.despine(fig)
fig.tight_layout()
fig.savefig(
	f'gaussian_dist.png', bbox_inches='tight', dpi=400
)
