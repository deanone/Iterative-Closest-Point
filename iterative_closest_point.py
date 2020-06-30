import numpy as np
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json


def load_points_json(p_json_filename, q_json_filename):
	
	# p point cloud
	f = open(p_json_filename)
	p_json = json.load(f)
	p_json = p_json['points']
	p = []
	for key, point in p_json.items():
		p.append(point)
	p = np.array(p)
	f.close()

	# q point cloud
	f = open(q_json_filename)
	q_json = json.load(f)
	q_json = q_json['points']
	q = []
	for key, point in q_json.items():
		q.append(point)
	q = np.array(q)
	f.close()

	min_shape = min(p.shape[0], q.shape[0])
	p = p[:min_shape, :]
	q = q[:min_shape, :]

	return p, q


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
		R_hat = np.matmul(VT.T, U.T)

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


def iterative_closest_point(p, q, low, high, max_iter = 50, tol = 1e-5):

	d = p.shape[1]

	p_hom = np.ones((d + 1, p.shape[0]))
	q_hom = np.ones((d + 1, q.shape[0]))
	p_hom[:d,:] = np.copy(p.T)
	q_hom[:d,:] = np.copy(q.T)

	prev_error = 0

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	fig.suptitle('Running ICP...')
	ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker = 'o')
	ax.scatter(q[:, 0], q[:, 1], q[:, 2], marker = '^')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_xlim(low, high)
	ax.set_ylim(low, high)
	ax.set_zlim(low, high)
	plt.draw()
	plt.pause(1)

	for i in range(max_iter):

		# find correspondence between the two cloud points
		nn = NearestNeighbors(n_neighbors = 1)
		nn_fit = nn.fit(q_hom[:d, :].T)
		distances, indices = nn_fit.kneighbors(p_hom[:d, :].T)

		# flatten the returned distances and indices arrays
		distances, indices = distances.ravel(), indices.ravel()
		
		# fit p cloud point to q cloud point
		T, R_hat, t_hat = fit_p_to_q(p_hom[:d, :].T, q_hom[:d, indices].T)

		# update the p cloud point
		p_hom = np.matmul(T, p_hom)

		# find error of current assignment
		current_error = np.mean(distances)

		print('Iteration {} - Mean Distance: {}'.format(i, round(current_error, 2)))

		p_new = p_hom.T[:,:d]
		q_new = q_hom.T[:,:d]

		ax.clear()
		ax.scatter(p_new[:, 0], p_new[:, 1], p_new[:, 2], marker = 'o')
		ax.scatter(q_new[:, 0], q_new[:, 1], q_new[:, 2], marker = '^')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_xlim(low, high)
		ax.set_ylim(low, high)
		ax.set_zlim(low, high)
		fig.canvas.draw_idle()
		plt.pause(1)

		if np.abs(current_error - prev_error) < tol:
			break

		prev_error = current_error

	p_new = p_hom.T[:,:d]

	return T, distances, p_new


def main():

	# for results reproducibility
	#np.random.seed(42)

	# number of points in point cloud
	#N = 10

	# dimension of the point cloud (e.g. 3D)
	#dim = 3

	# lower of the interval from which the random values of the coordinates of the points will be drawn
	#low = -1000

	# higher of the interval from which the random values of the coordinates of the points will be drawn
	#high = 1000

	# cloud points with (uniform) random coordinates
	#p = np.random.uniform(low, high, (N, dim))
	#q = np.random.uniform(low, high, (N, dim))

	p_json_filename = 'PointData1.json'
	q_json_filename = 'PointData2.json'
	p, q = load_points_json(p_json_filename, q_json_filename)
	low = min(np.min(p), np.min(q))
	high = max(np.max(p), np.max(q))

	# Run ICP algorithm (SVD-based variant)
	T, distances, p_new = iterative_closest_point(p, q, low, high, tol = 1e-10)

	print('\n')
	print('---------------------------------------------------')
	print('Final mean distance: ', round(np.mean(distances), 2))
	print('---------------------------------------------------')

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	fig.suptitle('ICP results')
	ax.scatter(p_new[:, 0], p_new[:, 1], p_new[:, 2], marker = 'o')
	ax.scatter(q[:, 0], q[:, 1], q[:, 2], marker = '^')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_xlim(low, high)
	ax.set_ylim(low, high)
	ax.set_zlim(low, high)

	plt.show()


if __name__ == '__main__':
	main()