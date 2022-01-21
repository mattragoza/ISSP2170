import numpy as np

v = np.array([
	[9, 5, 10]
])

u = np.array([
	[4],
	[1],
	[3]
])

A = np.array([
	[1, 2, 5],
	[3, 4, 6]
])

B = np.array([
	[7, 1, 9],
	[2, 2, 3],
	[4, 8, 6]
])

C = np.array([
	[ 8,  6, 5],
	[ 1, -3, 4],
	[-1, -2, 4]
])

print(u.T@u)
print(u@u.T)
print(v@u)
print(u+5)
print(A.T)
print(B@u)
print(np.linalg.inv(B))
print(B+C)
print(B-C)
print(A@B)
print(B@C)
print(B@A)
