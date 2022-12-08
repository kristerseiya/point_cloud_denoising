
import numpy as np
from scipy.sparse.linalg import lobpcg as LOBPCG
from scipy.sparse.linalg import lsqr as LSQR
from scipy import sparse
import open3d as o3d
import scipy.spatial

import geometry

def _project_M_diagonal_constraint(M, max_trace):

    m_diag = np.diag(M)
    m_thres = np.sum(np.abs(M - np.diag(m_diag)), axis=1)
    K = M.shape[0]
    n = K

    m_diag = m_diag - (np.mean(m_diag) - max_trace / n)
    below_thres = (m_diag < m_thres)
    unthreshed = np.ones((K), dtype=bool)

    while np.any(below_thres):
        m_diag[below_thres] = m_thres[below_thres]
        unthreshed = unthreshed * (~below_thres)
        n = np.sum(unthreshed)
        m_diag = m_diag - (np.mean(m_diag[unthreshed]) - max_trace / n) * unthreshed
        below_thres = (m_diag < m_thres)

    M[range(K), range(K)] = m_diag

    return M


def optimize_M_diagonal(init_M, feature_diff, point_dist, max_trace, step_size, n_iter):

    M = np.copy(init_M)
    K = M.shape[0]
    feature_diff_2 = feature_diff**2
    tmp1 = (feature_diff_2.T * point_dist)

    for it in range(n_iter):
        # print('diagonal {:d}/{:d}'.format(it2+1,n_pg_iter))
        m_diag = np.diag(M)
        m_grad = - tmp1 @ np.exp( - feature_diff_2 @ m_diag )
        m_diag = m_diag - step_size * m_grad
        M[range(K), range(K)] = m_diag
        M = _project_M_diagonal_constraint(M, max_trace)
        # print(m_diag)

    return M

def _compute_minimum_eigenvalue(M_sub2):
    min_eig_val, _ = LOBPCG(M_sub2, np.random.randn(M_sub2.shape[0],1), largest=False)
    return min_eig_val[0]

def _project_M_offdiagonal_constraint(m_col_offdiag, M_sub1, M_sub2):
    m_col_offdiag_norm = np.linalg.norm(m_col_offdiag, ord=2)
    min_eig_val = _compute_minimum_eigenvalue(M_sub2)
    norm_thres = np.sqrt(min_eig_val*M_sub1)
    if m_col_offdiag_norm <= norm_thres:
        return m_col_offdiag
    return m_col_offdiag / m_col_offdiag_norm * norm_thres

def _get_block(M, idx):
    M_sub = np.zeros((len(idx), len(idx)), dtype=M.dtype)
    for i, idx1 in enumerate(idx):
        for j, idx2 in enumerate(idx):
            M_sub[i,j] = M[idx1, idx2]
    return M_sub


def optimize_M_offdiagonal(init_M, feature_diff, point_dist, max_trace, step_size, n_block_iter, n_pg_iter):

    M = np.copy(init_M)
    K = M.shape[0]
    feature_diff_2 = feature_diff**2

    for it1 in range(n_block_iter):

        for k in range(K):

            not_k = np.concatenate([range(0,k), range(k+1, K)])
            not_k = not_k.astype(int)
            M_sub1 = M[k,k]
            M_sub2 = _get_block(M, not_k)
            feature_diff_sub1 = feature_diff[:,k]
            feature_diff_sub2 = feature_diff[:,not_k]
            m_col_offdiag = M[not_k, k]
            point_dist_sub = np.exp( - (feature_diff_sub1**2) * M_sub1 -  np.sum((feature_diff_sub2 @ M_sub2) * feature_diff_sub2, axis=1) ) * point_dist
            tmp1 =  -2 * (feature_diff_sub2.T * feature_diff_sub1 * point_dist_sub)

            for it2 in range(n_pg_iter):
                # print('off_diagonal column {:d}/{:d}'.format(it2+1,n_pg_iter))
                grad = tmp1 @ np.exp( -2 * feature_diff_sub1 * (feature_diff_sub2 @ m_col_offdiag) )
                m_col_offdiag = m_col_offdiag - step_size * grad
                m_col_offdiag = _project_M_offdiagonal_constraint(m_col_offdiag, M_sub1, M_sub2)
                # print(m_col_offdiag)

            M[not_k, k] = m_col_offdiag
            M[k, not_k] = m_col_offdiag

    return M


def denoise_point_cloud(noisy_pcl, n_patch, patch_size, n_patch_neighbor,
                        n_iter, optim_M_iter, m_offdiag_n_block_iter, m_offdiag_n_pg_iter, m_diag_n_pg_iter,
                        m_offdiag_pg_step_size, m_diag_pg_step_size,
                        max_trace=3):

    points = np.array(noisy_pcl.points)
    M = np.eye(6) * max_trace / 6

    for it in range(n_iter):
        print('{:d}/{:d}'.format(it+1,n_iter))

        # gamma= 0.2**(it+1) * (np.exp(1)-1)**(1-it-1)
        gamma= 0.2**(it/2+1) * (np.exp(1)-1)**(1-(it+1)/2)

        pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcl.estimate_normals()
        pcl.orient_normals_consistent_tangent_plane(10)

        features = np.concatenate([points, pcl.normals], axis=1)
        point_knn = geometry.compute_knn(points, patch_size)
        n_point = len(points)

        dpcl_idx = geometry.farthest_point_sampling(points, n_patch)
        # dpcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[dpcl_idx]))
        patch_knn = geometry.compute_knn(points[dpcl_idx], n_patch)

        sample_idx = np.zeros((n_patch, patch_size), dtype=int)                       # point index stored in patch index order
        all_patches = np.zeros((n_patch, patch_size, 3))                              # patches stored in patch index order
        feature_diff = np.zeros((n_patch, patch_size, n_patch_neighbor, 6))           # feature difference of pairs stored in patch index order
        point_dist = np.zeros((n_patch, patch_size, n_patch_neighbor))                # point distance of pairs stored in patch index order
        corres_idx = np.zeros((n_patch, patch_size, n_patch_neighbor), dtype=int)     # point index of pair stored in patch index order
        points_graph = np.zeros((n_patch, patch_size, n_patch_neighbor), dtype=int)   # patch index of pair store in patch index order

        for i in range(n_patch):
            for j in range(patch_size):
                sample_idx[i,j] = point_knn[dpcl_idx[i], j]
                all_patches[i,j] = points[sample_idx[i,j]]

        for i in range(n_patch):
            patch = all_patches[i]
            for k in range(n_patch_neighbor):
                near_patch_idx = patch_knn[i,k]
                near_patch = all_patches[near_patch_idx]
                for j in range(patch_size):
                    point = patch[j]
                    idx = np.argmin(np.linalg.norm(near_patch - point, ord=2, axis=1))
                    corres_idx[i,j,k] = sample_idx[near_patch_idx, idx]
                    feature_diff[i,j,k] = features[sample_idx[i,j]] - features[corres_idx[i,j,k]]
                    point_dist[i,j,k] = points[sample_idx[i,j]] @ points[corres_idx[i,j,k]]
                    points_graph[i, j, k] = near_patch_idx*patch_size+idx

        feature_diff_flat = np.reshape(feature_diff, (-1, 6))
        point_dist_flat = np.reshape(point_dist, (-1))

        print('start optimize M')
        for it in range(optim_M_iter):
            M = optimize_M_offdiagonal(M, feature_diff_flat, point_dist_flat, max_trace, m_offdiag_pg_step_size, m_offdiag_n_block_iter, m_offdiag_n_pg_iter)
            M = optimize_M_diagonal(M, feature_diff_flat, point_dist_flat, max_trace, m_diag_pg_step_size, m_diag_n_pg_iter)

        W_row_idx = np.kron( range(n_patch*patch_size), np.ones((n_patch_neighbor),dtype=int) )
        W_col_idx = np.reshape(points_graph, (-1))
        W_data = np.exp( - np.sum((feature_diff_flat @ M) * feature_diff_flat, axis=1) )
        W = sparse.csr_matrix((W_data, (W_row_idx, W_col_idx)), shape=(n_patch*patch_size, n_patch*patch_size))

        D = sparse.diags(W @ np.ones(n_patch*patch_size), 0)
        L = D - W

        S_row_idx = range(n_patch*patch_size)
        S_col_idx = np.reshape(sample_idx, (-1))
        S_data = np.ones((n_patch*patch_size))
        S = sparse.csr_matrix((S_data, (S_row_idx, S_col_idx)), shape=(n_patch*patch_size, n_point))

        C = np.kron(points[dpcl_idx], np.ones((patch_size,1)))

        StL = S.T @ L
        StLS = StL @ S
        Ax = gamma * StLS + sparse.diags(np.ones((n_point)), 0)
        bx = points[:,0] + gamma * StL @ C[:,0]
        Ay = gamma * StLS + sparse.diags(np.ones((n_point)), 0)
        by = points[:,1] + gamma * StL @ C[:,1]
        Az = gamma * StLS + sparse.diags(np.ones((n_point)), 0)
        bz = points[:,2] + gamma * StL @ C[:,2]

        points_x = LSQR(Ax, bx)[0]
        points_y = LSQR(Ay, by)[0]
        points_z = LSQR(Az, bz)[0]

        points = np.stack([points_x, points_y, points_z], axis=1)

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def denoise_point_cloud_2(noisy_points, n_patch, patch_size, n_patch_neighbor,
                        n_iter, optim_M_iter, m_offdiag_n_block_iter, m_offdiag_n_pg_iter, m_diag_n_pg_iter,
                        m_offdiag_pg_step_size, m_diag_pg_step_size,
                        max_trace=3, gamma=None, ground_truth=None):

    points = np.copy(noisy_points)
    n_point = len(points)
    normals = np.random.randn(*points.shape)
    M = np.eye(6) * max_trace / 6

    if gamma is None:
        gamma = 0.2**(np.arange(n_iter)+1) * (np.exp(1)-1)**(1-np.arange(n_iter))
    if np.isscalar(gamma):
        gamma = np.array([gamma]*n_iter)
    gamma_arr = gamma

    if ground_truth is not None:
        grouth_truth_kdtree = scipy.spatial.cKDTree(ground_truth, leafsize=16)

    for it in range(n_iter):

        print('{:d}/{:d}'.format(it+1,n_iter))
        # gamma= 0.2**(it+1) * (np.exp(1)-1)**(1-it-1)
        # gamma= 0.1**(it/2+1) * (np.exp(1)-1)**(1-(it+1)/2)
        gamma = gamma_arr[it]

        print('computing normals')
        point_kdtree = scipy.spatial.cKDTree(points)
        neighbor_idx, _, _ = geometry.compute_neighbors(points, patch_size-1, kdtree=point_kdtree)
        normals = geometry.compute_normals(points, neighbor_idx, init_normals=normals)

        print('sampling for patch center points')
        patch_idx = geometry.farthest_point_sampling(points, n_patch)
        patch_neighbor_idx, _, _ = geometry.compute_neighbors(points[patch_idx], n_patch_neighbor)

        print('getting ready for optimization of M')
        sample_idx = np.zeros((n_patch, patch_size), dtype=int)                       # point index stored in patch index order
        all_patches = np.zeros((n_patch, patch_size, 3))                              # patches stored in patch index order
        feature_diff = np.zeros((n_patch, patch_size, n_patch_neighbor, 6))           # feature difference of pairs stored in patch index order
        point_dist = np.zeros((n_patch, patch_size, n_patch_neighbor))                # point distance of pairs stored in patch index order
        # corres_idx = np.zeros((n_patch, patch_size, n_patch_neighbor), dtype=int)     # point index of pair stored in patch index order
        sample_graph = np.zeros((n_patch, patch_size, n_patch_neighbor), dtype=int)   # patch index of pair store in patch index order

        for i in range(n_patch):
            sample_idx[i,0] = patch_idx[i]
            all_patches[i,0] = points[patch_idx[i]]
            for j in range(1, patch_size):
                sample_idx[i,j] = neighbor_idx[patch_idx[i], j-1]
                all_patches[i,j] = points[sample_idx[i,j]] - points[sample_idx[i,0]]

        for i in range(n_patch):
            patch = all_patches[i]
            for k in range(n_patch_neighbor):
                near_patch_idx = patch_neighbor_idx[i,k]
                near_patch = all_patches[near_patch_idx]
                for j in range(patch_size):
                    point = patch[j]
                    idx = np.argmin(np.linalg.norm(near_patch - point, ord=2, axis=1))
                    corres_idx = sample_idx[near_patch_idx, idx]
                    feature_diff[i,j,k,:3] = points[sample_idx[i,j]] - points[corres_idx]
                    normal_dir = np.sign(normals[sample_idx[i,j]] @ normals[corres_idx])
                    feature_diff[i,j,k,3:] = normals[sample_idx[i,j]] - normal_dir * normals[corres_idx]
                    point_dist[i,j,k] = np.linalg.norm(near_patch[idx] - point, ord=2)**2
                    sample_graph[i,j,k] = near_patch_idx*patch_size+idx

        feature_diff_flat = np.reshape(feature_diff, (-1, 6))
        point_dist_flat = np.reshape(point_dist, (-1))

        print('start optimizing for M')
        for it in range(optim_M_iter):
            M = optimize_M_offdiagonal(M, feature_diff_flat, point_dist_flat, max_trace, m_offdiag_pg_step_size, m_offdiag_n_block_iter, m_offdiag_n_pg_iter)
            M = optimize_M_diagonal(M, feature_diff_flat, point_dist_flat, max_trace, m_diag_pg_step_size, m_diag_n_pg_iter)

        print('contructing sparse matrices')
        W_row_idx = np.kron( range(n_patch*patch_size), np.ones((n_patch_neighbor),dtype=int) )
        W_col_idx = sample_graph.flatten()
        W_data = np.exp( - np.sum((feature_diff_flat @ M) * feature_diff_flat, axis=1) )
        W = sparse.csr_matrix((W_data, (W_row_idx, W_col_idx)), shape=(n_patch*patch_size, n_patch*patch_size))

        D = sparse.diags(W @ np.ones(n_patch*patch_size), 0)
        L = D - W

        S_row_idx = range(n_patch*patch_size)
        S_col_idx = sample_idx.flatten()
        S_data = np.ones((n_patch*patch_size))
        S = sparse.csr_matrix((S_data, (S_row_idx, S_col_idx)), shape=(n_patch*patch_size, n_point))

        C = np.kron(points[patch_idx], np.ones((patch_size,1)))

        StL = S.T @ L
        StLS = StL @ S
        Ax = gamma * StLS + sparse.diags(np.ones((n_point)), 0)
        bx = points[:,0] + gamma * StL @ C[:,0]
        Ay = gamma * StLS + sparse.diags(np.ones((n_point)), 0)
        by = points[:,1] + gamma * StL @ C[:,1]
        Az = gamma * StLS + sparse.diags(np.ones((n_point)), 0)
        bz = points[:,2] + gamma * StL @ C[:,2]

        print('solve for points')
        points_x = LSQR(Ax, bx)[0]
        points_y = LSQR(Ay, by)[0]
        points_z = LSQR(Az, bz)[0]

        points = np.stack([points_x, points_y, points_z], axis=1)

        if ground_truth is not None:
            error_distance = geometry.compute_distance_between_points(grouth_truth_kdtree, points, ref_is_kdtree=True)
            error_distance = np.mean(error_distance)
            print('MSE: {:.3e}'.format(error_distance))


    return points


def my_denoise_algo(noisy_points, n_iter, sample_rate, n_neighbor, sigma_d, sigma_n, lambd, ground_truth=None):

    noisy_point_kdtree = scipy.spatial.cKDTree(noisy_points)
    points = np.copy(noisy_points)
    n_point = points.shape[0]
    normals = np.random.randn(*points.shape)
    n_sample = int(sample_rate * n_point)
    lambd_arr = 50*np.exp(-np.linspace(0,50,101))
    # lambd_arr = np.array([50]*101)

    if ground_truth is not None:
        grouth_truth_kdtree = scipy.spatial.cKDTree(ground_truth, leafsize=16)

    for it in range(n_iter):

        print('{:d}/{:d}'.format(it+1, n_iter))
        directions = np.zeros((n_point, 3))
        neighbor_idx = np.zeros((n_point, n_neighbor), dtype=int)
        neighbor_dist = np.zeros((n_point, n_neighbor), dtype=int)
        neighbor_count = np.zeros((n_point), dtype=int)
        total_weights = np.zeros((n_point))

        print('building kdtree')
        point_kdtree = scipy.spatial.cKDTree(points)

        print('computing normals')
        # neighbor_idx, neighbor_dist, neighbor_count = geometry.compute_neighbors(points, n_neighbor, kdtree=point_kdtree)
        # normals = geometry.compute_normals(points, neighbor_idx, init_normals=normals)
        for i, i_point in enumerate(points):
            neighbor_info = point_kdtree.query(i_point, n_neighbor+1)
            neighbor_idx[i] = neighbor_info[1][1:]
            neighbor_dist[i] = neighbor_info[0][1:]
            neighbor_count[neighbor_idx[i]] = neighbor_count[neighbor_idx[i]] + 1
            neighbor_points = np.concatenate([ [i_point], points[neighbor_idx[i]], 2*i_point - points[neighbor_idx[i]]  ], axis=0)
            # neighbor_mu = np.mean(neighbor_points, axis=0)
            neighbor_mu = np.copy(i_point)
            X = neighbor_points - neighbor_mu
            _, eigvecs = LOBPCG(X.T @ X, np.reshape(normals[i], (3,1)), largest=False)
            # _, eigvecs = LOBPCG(X.T @ X, np.random.randn(3,1), largest=False)
            normals[i] = eigvecs[:,0] / np.linalg.norm(eigvecs[:,0], ord=2)

        sample_idx = np.random.choice(range(n_point), size=(n_sample), replace=False)
        for i, i_sample_idx in enumerate(sample_idx):
            i_point = points[i_sample_idx]
            i_normal = normals[i_sample_idx]
            i_neighbor_idx = neighbor_idx[i_sample_idx]
            neighbor_points = points[i_neighbor_idx]
            center_point = np.mean(np.concatenate([[i_point], neighbor_points], axis=0))
            center_point = np.copy(i_point)
            neighbor_normals = normals[i_neighbor_idx]
            neighbor_normal_cos = neighbor_normals @ i_normal
            neighbor_normal_dist = np.sqrt( np.abs(2 - 2 * np.abs(neighbor_normal_cos)) )
            neighbor_plane_dist = (neighbor_points - center_point) - np.expand_dims((neighbor_points - center_point) @ i_normal, axis=1) * np.expand_dims(i_normal, axis=0)
            neighbor_plane_dist = np.linalg.norm(neighbor_plane_dist, ord=2, axis=1)
            # neighbor_weights = np.exp( - neighbor_plane_dist**2 / sigma_d**2 / 2 - neighbor_normal_dist**2 / sigma_n**2 / 2 )
            # neighbor_weights = np.exp( - neighbor_plane_dist**2 / sigma_d**2 / 2)
            # neighbor_weights = np.exp( - neighbor_dist[i_sample_idx]**2 / sigma_d**2 / 2)
            i_neighbor_ortho_dist = np.abs( (neighbor_points - center_point) @ i_normal )
            i_neighbor_sigma = np.linalg.norm(neighbor_points-center_point, ord=2, axis=1) * sigma_d + 1e-9
            neighbor_weights = 1 / i_neighbor_sigma * np.exp( - 0.5 * (i_neighbor_ortho_dist / i_neighbor_sigma)**2 )
            # neighbor_weights = np.ones((n_neighbor))
            total_weights[i_neighbor_idx] += neighbor_weights
            directions[i_neighbor_idx] += np.expand_dims( neighbor_weights * ( (neighbor_points - center_point) @ i_normal ), axis=1) * np.expand_dims(i_normal, axis=0)
            # point_plane_dist = (i_point - center_point) - np.sum((i_point - center_point) * i_normal) * i_normal
            # point_plane_dist = np.linalg.norm(point_plane_dist, ord=2)
            # i_weight = np.exp( - point_plane_dist**2 / sigma_d**2 / 2 )
            # directions[i_sample_idx] = directions[i_sample_idx] + i_weight * np.sum( (i_point - center_point) * i_normal ) * i_normal

        #nonzero_neighbor = neighbor_count != 0
        #directions[nonzero_neighbor] = directions[nonzero_neighbor] / np.expand_dims(neighbor_count[nonzero_neighbor], axis=1)
        nonzero_weights = (total_weights != 0)
        directions[nonzero_weights] /= np.expand_dims(total_weights[nonzero_weights], axis=1)
        # directions = directions / np.expand_dims(neighbor_count+1, axis=1)
        # points = points - directions

        lambd = lambd_arr[it]
        print(lambd)
        noisy_patch_mean = np.zeros((n_point, 3))
        for i in range(n_point):
            i_point = points[i]
            noisy_patch_idx = noisy_point_kdtree.query(i_point, 10)[1]
            noisy_patch_mean[i] = np.mean(noisy_points[noisy_patch_idx], axis=0)
        # points = (noisy_points + lambd * (points - directions)) / (1 + lambd)
        points = (noisy_patch_mean + lambd * (points - directions)) / (1 + lambd)

        if ground_truth is not None:
            error_distance = geometry.compute_distance_between_points(grouth_truth_kdtree, points, ref_is_kdtree=True)
            error_distance = np.mean(error_distance)
            print('MSE: {:.3e}'.format(error_distance))

    return points
