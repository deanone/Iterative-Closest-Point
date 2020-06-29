import numpy as np


def fit_p_to_q(p, q):

	if p.shape == q.shape:

		(N, d) = p.shape

		# Normalize clould points by subtracting for each of them their mean points
		mean_p = np.mean(p, axis = 0)
		mean_q = np.mean(q, axis = 0)
		p_norm = p - mean_p
		q_norm = q - mean_q

		# Correlation matrix
		H = np.zeros((d, d))
		for i in range(N):
			x = p_norm[i, :]
			x = x.reshape((d, 1))

			y = q_norm[i, :]
			y = y.reshape((1, d))

			H += np.matmul(x, y)

		# SVD of H
		U, sigma, VT = np.linalg.svd(H, full_matrices = True)

		# Rotation matrix
		R_hat = np.matmul(np.transpose(VT), np.transpose(U))

		# Translation vector
		#T_hat = mean_q - np.matmul(R_hat, mean_p)

		t_hat = mean_q.T - np.matmul(R_hat, mean_p.T)

		# Homogeneous transformation
		T = np.identity(d + 1)
		T[:d, :d] = R_hat
		T[:d, d] = t_hat

		return T, R_hat, t_hat

	else:

		R_hat = np.zeros((d, d))
		t_hat = np.zeros(d)
		T = np.identity(d + 1)
		T[:d, :d] = R_hat
		T[:d, d] = t_hat
		return T, R_hat, t_hat


def main():

	# For results reproducibility
	np.random.seed(42)

	# Number of points in point cloud
	N = 10

	# Dimension of the point cloud (e.g. 3D)
	dim = 3

	# lower of the interval from which the random values of the coordinates of the points will be drawn
	low = 0

	# higher of the interval from which the random values of the coordinates of the points will be drawn
	high = 10

	# Cloud points with (uniform) random coordinates
	p = np.random.normal(low, high, (N, dim))
	q = np.random.normal(low, high, (N, dim))

	# Run ICP algorithm (SVD-based variant)
	T, R_hat, t_hat = fit_p_to_q(p, q)



if __name__ == '__main__':
	main()