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
data = pd.read_csv('coin.txt')
data.columns = [c.strip() for c in data.columns]

# count heads and tails
n1 = sum(data.outcome == 1)
n2 = sum(data.outcome == 0)
print(f'N1 = {n1}, N2 = {n2}, N = {n1+n2}')

# compute MLE of Bernoulli parameter
def bernoulli_mle(n1, n2):
	return n1 / (n1 + n2)

theta_mle = bernoulli_mle(n1, n2)
print(f'theta_MLE = {theta_mle}')

# compute MAP of Bernoulli parameter
def bernoulli_map(n1, n2, a1, a2):
	return (n1 + a1 - 1) / (n1 + n2 + a1 + a2 - 2)

a1, a2 = 1, 1 # set prior counts
print(f'\nprior = Beta(theta|alpha1={a1},alpha2 = {a2})')
theta_map = bernoulli_map(n1, n2, a1, a2)
print(f'theta_MAP = {theta_map}')

# compute expected theta given data
theta_exp = scipy.stats.beta.mean(a1 + n1, a2 + n2)
print(f'E[theta|D] = {theta_exp}')

# plot prior and posterior of Bernoulli parameter
fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))

def plot_prior_post(n1, n2, a1, a2, ax):

	colors = sns.color_palette()
	x = np.linspace(0, 1, 1000)
	y1 = scipy.stats.beta.pdf(x, a1, a2)
	y2 = scipy.stats.beta.pdf(x, n1 + a1, n2 + a2)
	ax.fill_between(x, 0, y1, alpha=0.25)
	ax.fill_between(x, 0, y2, alpha=0.25)
	ax.plot(x, y1, label='prior')
	ax.plot(x, y2, label='posterior')

	p_theta = scipy.stats.beta.pdf(theta_map, n1 + a1, n2 + a2)
	y_min, y_max = ax.get_ylim()
	ax.vlines(theta_map, y_min, p_theta, color=colors[1])
	ax.vlines(theta_exp, y_min, p_theta, color=colors[1])
	ax.set_ylim(0, 10)
	ax.set_xlim(0, 1)
	xticks = list(np.arange(0, 1.01, 0.25))
	xticklabels = [f'{xt:.2f}' for xt in xticks]
	ax.set_xticks(xticks + [theta_map, theta_exp])
	ax.set_xticklabels(
		xticklabels + [r'$\theta_{MAP}$', '\n' + r'$E[\theta|D]$']
	)
	ax.legend(frameon=False)
	ax.set_xlabel(r'$\theta$')
	ax.set_ylabel(r'$P(\theta)$')
	ax.set_title('Prior = ' + f'$Beta(\\theta|\\alpha_1={a1},\\alpha_2={a2})$')
	ax.set_axisbelow(True) # grid lines behind data
	ax.grid(True, linestyle=':', color='lightgray')

plot_prior_post(n1, n2, a1, a2, ax=axes[0])

# repeat with more informative prior

a1, a2 = 4, 2 # set prior counts
print(f'\nprior = Beta(theta|alpha1={a1},alpha2 = {a2})')
theta_map = bernoulli_map(n1, n2, a1, a2)
print(f'theta_MAP = {theta_map}')

# compute expected theta given data
theta_exp = scipy.stats.beta.mean(a1 + n1, a2 + n2)
print(f'E[theta|D] = {theta_exp}')

plot_prior_post(n1, n2, a1, a2, ax=axes[1])

sns.despine(fig)
fig.tight_layout()
fig.savefig(
	f'coin_flip_prior_post.png', bbox_inches='tight', dpi=400
)
