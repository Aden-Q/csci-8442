from kd_tree import KDTree

# test the KDTree
points_list = [[5, 4],
                [2, 3],
                [8, 1],
                [9, 6],
                [7, 2],
                [4, 7]]

kd_tree = KDTree(points_list, 2)
assert(kd_tree.root.data == [7, 2])

query_point = [3, 4.5]
assert(kd_tree.get_knn(query_point, 2) == [[5, 4], [2, 3]])
assert(kd_tree.get_nn(query_point) == [2, 3])
