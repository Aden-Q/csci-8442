from pa_tree import PATree
import numpy as np

# test the PATree
points_list = np.array([[5, 4, 3],
                   [2, 3, 5],
                   [8, 1, 2],
                   [9, 6, 1],
                   [7, 2, 5],
                   [4, 7, 6]])

pa_tree = PATree(points_list, 2)
assert(np.array_equal(pa_tree.root.data, np.array([7, 2, 5])))

query_point = [3, 4.5]
assert(np.array_equal(pa_tree.get_knn(query_point, 2), np.array([np.array([5, 4, 3]), np.array([2, 3, 5])])))
assert(np.array_equal(pa_tree.get_nn(query_point), np.array([2, 3, 5])))
