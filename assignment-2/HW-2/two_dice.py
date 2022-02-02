import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set_palette('pastel')
np.random.seed(1)

fig_w = 3
fig_h = 1.5
y_max = 175

# get all possible outcomes of two dice rolls
one_dice = np.arange(1, 7)
two_dice = one_dice[:,np.newaxis] + one_dice[np.newaxis,:]

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
sns.histplot(
	two_dice.flatten(),
	discrete=True,
	stat='density'
)
sns.despine(fig)
ax.set_xlim(two_dice.min()-1, two_dice.max()+1)
ax.set_xlabel('$X$')
ax.set_ylabel('$P(X)$')
ax.set_title('Sum of two dice')
ax.set_axisbelow(True) # grid lines behind data
ax.grid(True, linestyle=':', color='lightgray')
fig.savefig('two_dice_prob.png', bbox_inches='tight', dpi=400)

counts, edges = np.histogram(
	two_dice.flatten(), bins=np.arange(2, 14, 1)
)
print(edges)
print(counts)
print(counts.sum())
print(counts/counts.sum())
