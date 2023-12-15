from numba import njit
from route_operations import *


@njit
def test_neighbor(path):
    for i in range(4):
        get_neighbor(path, i)
    print("test_neighbor passed")
test_neighbor([*range(15)])
