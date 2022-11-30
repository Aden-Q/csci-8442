# An implementation of a kd-tree

import heapq
import numpy as np
from sklearn.decomposition import PCA

class Node:
    '''A node in a PCA-Tree'''
    def __init__(self, data=None, dim=None, left=None, right=None):
        '''
        Args:
        data: The data stored in the node.
        dim: The dimension to split the dataset.
        left: The left child of the node.
        right: The right child of the node.
        '''
        self.data = data
        self.dim = dim
        self.left = left
        self.right = right

class HeapNode:
    '''A node in a min-heap'''
    def __init__(self, data, dist):
        self.data = data
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist

class PCATree:
    '''
    Usage:
    1. Create a PCATree object with the given points.
        `kd_tree = KDTree(point_list, dim, dist_func)`
    2. Search for the k nearest neighbors of a given point.
        `kd_tree.get_knn(point, k)`
    3. Search for the nearest neighbor of a given point.
        `kd_tree.get_nn(point)`
    '''
    def __init__(self, points_list, max_dim, dist_func=None) -> None:
        '''Initialize the KDTree with the given poitns.
        Args:
        point_list: A list of points to be stored in the tree, a numpy array of num_points x dimension.
        max_dim: The maximal dimension of the points.
        dist_func: A function that takes two points and returns the distance
        '''
        self.pca = PCA(n_components=1, svd_solver='full')

        def build_tree(points_list, node, dim, max_dim):
            '''Utile function to build the tree recursively.
            Args:
            points_list: A list of points to be stored in the tree, a numpy array of num_points x dimension.
            node: The current node.
            dim: The current dimension.
            max_dim: The maximal dimension of the points.
            Returns: A node in the Kd tree.
            '''
            # sort the points by the current dimension
            vectors = np.array(sorted(points_list, key=lambda x: x[dim]))
            if len(vectors) == 0:
                # if there is no point, return None

                return None
            if len(vectors) == 1:
                # if there is only one point, return a leaf node
                node.data = vectors[0]
                node.dim = dim

                return node
            if len(vectors) == 2:
                node.data = vectors[1]
                node.dim = dim
                node.left = Node(data=vectors[0], dim=(dim + 1) % max_dim)

                return node
            # if there are more than two points, split the points into two part
            # and recursively build the tree
            node.dim = dim
            vectors_pca = self.pca.fit_transform(vectors)
            median = np.median(vectors_pca)
            left_subsets = vectors[vectors_pca[:, 0] < median]
            right_min = np.min(vectors_pca[vectors_pca >= median])
            right_subsets = vectors[vectors_pca[:, 0] > right_min]
            node.data = vectors[vectors_pca[:, 0] == right_min, :][0]
            node.left = build_tree(left_subsets, Node(), (dim+1) % max_dim, max_dim)
            node.right = build_tree(right_subsets, Node(), (dim+1) % max_dim, max_dim)

            return node

        self.root = Node()
        self.root = build_tree(points_list, self.root, 0, max_dim)
        # Default distance function is Euclidean distance
        if dist_func is None:
            dist_func = lambda x, y: sum((a - y[i]) ** 2 for i, a in enumerate(x))
        self.dist_func = dist_func

    def _get_knn(self, node, point, k, heap) -> list:
        '''Utility function to search for the k nearest neighbors of a given point.
        Args:
        node: The current node.
        point: The query point.
        k: The number of nearest neighbors to be returned.
        heap: A heap to store the k nearest neighbors.
        Returns:

        '''
        if node is None:
            return []
        # calculate the distance between the query point and the current node
        dist = self.dist_func(point, node.data)
        # if the heap is not full, push the current node into the heap
        if len(heap) < k:
            heapq.heappush(heap, HeapNode(node.data, -dist))
        else:
            # if the heap is full, only push the current node into the heap if
            # the distance is smaller than the largest distance in the heap
            if dist < -heap[0].dist:
                heapq.heappushpop(heap, HeapNode(node.data, -dist))
        # if the query point is on the left side of the current node, search the left subtree first
        if point[node.dim] < node.data[node.dim]:
            self._get_knn(node.left, point, k, heap)
            # if the distance between the query point and the hyperplane
            # is smaller than the largest distance in the heap, search the right subtree
            if abs(point[node.dim] - node.data[node.dim]) < -heap[0].dist:
                self._get_knn(node.right, point, k, heap)
        else:
            self._get_knn(node.right, point, k, heap)
            if abs(point[node.dim] - node.data[node.dim]) < -heap[0].dist:
                self._get_knn(node.left, point, k, heap)

        return heap

    def get_knn(self, point, k) -> list:
        '''Search for the k nearest neighbors of a given point.
        Args:
        point: The query point.
        k: The number of nearest neighbors to be returned.
        Returns:
        A list of the k nearest neighbors. (data, label)
        '''
        return [x.data for x in self._get_knn(self.root, point, k, [])]
        
    def get_nn(self, point):
        '''Search for the nearest neighbor of a given point.
        Args:
        point: The query point.
        Returns:

        '''
        return [x.data for x in self._get_knn(self.root, point, 1, [])][0]


if __name__ == '__main__':
    # test the PCATree
    points_list = np.array([[5, 4, 3],
                   [2, 3, 5],
                   [8, 1, 2],
                   [9, 6, 1],
                   [7, 2, 5],
                   [4, 7, 6]])

    pca_tree = PCATree(points_list, 2)
    print(pca_tree.root.data)

    query_point = [3, 4.5]
    print(pca_tree.get_knn(query_point, 2))
    print(pca_tree.get_nn(query_point))
