#pylint: disable=no-name-in-module
from scipy.spatial import cKDTree

class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN

        inp: input data, single frame or multi frame

        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []
            # print("Search Loops", len(inp.T))
            # print((inp.T))
            # print()
            for i in inp.T:
                i_dist, i_index = self.tree.query(i, k=k)
                index.append(i_index)
                dist.append(i_dist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index

