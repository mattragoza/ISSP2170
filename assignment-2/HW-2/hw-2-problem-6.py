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
fig_h = 4

# load data set
data = pd.read_csv('poisson.txt')
data.columns = [c.strip() for c in data.columns]
print(data.describe())
x = data.value

# MLE of Poisson lambda
sum_x = sum(x)
n = len(x)
lambda_mle = sum_x / n
print(f'lambda_MLE = {lambda_mle}')

# plot some poisson distributions

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
colors = sns.color_palette()
x_min = 0
x_max = 14
x = np.arange(x_min, x_max+1)

lambda1 = 2
lambda2 = 6
y1 = scipy.stats.poisson.pmf(x, lambda1)
y2 = scipy.stats.poisson.pmf(x, lambda2)
y3 = scipy.stats.poisson.pmf(x, lambda_mle)

for i, (y, lam) in enumerate([(y1, lambda1), (y2, lambda2), (y3, lambda_mle)]):
	ax = axes[i]
	r, g, b = colors[i]
	width = 1.0
	ax.bar(x, y,
		color=(r,g,b,0.25),
		edgecolor=(r,g,b,1.0),
		label=f'$\\lambda = {lam:.2f}$',
		width=width, linewidth=1, 
	)

	ax.set_ylabel('$P(X)$')
	ax.set_xlabel('$X$')
	ax.set_title(f'$Poisson(x|\\lambda={lam})$')
	#ax.legend(frameon=False, title='')

	ax.set_xticks(np.arange(0, 15, 4))
	ax.set_xlim(x_min-width/2, x_max+width/2)
	ax.set_ylim(0, ax.get_ylim()[1])
	ax.set_axisbelow(True) # grid lines behind data
	ax.grid(True, linestyle=':', color='lightgray')

sns.despine(fig)
fig.tight_layout()
fig.savefig(
	f'poisson_dist.png', bbox_inches='tight', dpi=400
)

# now plot the gamma distributions

fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))
colors = sns.color_palette()
x_min = 0
x_max = 14
x = np.linspace(x_min, x_max, 1000)


for i, (a, b) in enumerate([(1, 2), (3, 5)]):
	ax = axes[i]

	print(f'prior = Gamma(lambda|{a},{b})')

	# plot prior
	prior = scipy.stats.gamma(a=a, scale=b)
	y = prior.pdf(x)
	ax.fill_between(x, 0, y, alpha=0.25)
	ax.plot(x, y, label='prior')

	lambda_prior = prior.mean()
	print(f'E[lambda] = {lambda_prior}')

	# plot posterior
	posterior = scipy.stats.gamma(a=a+sum_x, scale=b/(n*b+1))
	y = posterior.pdf(x)
	ax.fill_between(x, 0, y, alpha=0.25)
	ax.plot(x, y, label='posterior')

	lambda_post = posterior.mean()
	print(f'E[lambda|D] = {lambda_post}')

	ax.set_ylabel('$P(\\lambda)$')
	ax.set_xlabel('$\\lambda$')
	ax.set_title('Prior = ' + f'$Gamma(\\lambda|a={a},b={b})$')
	ax.legend(frameon=False, title='')

	ax.set_xlim(x_min, x_max)
	ax.set_ylim(0, ax.get_ylim()[1])
	ax.set_axisbelow(True) # grid lines behind data
	ax.grid(True, linestyle=':', color='lightgray')

sns.despine(fig)
fig.tight_layout()
fig.savefig(
	f'gamma_dist.png', bbox_inches='tight', dpi=400
)
