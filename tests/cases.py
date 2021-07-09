import scipy.io
from dlopy.dlo import Soil, DLO, calc
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
                           assert_array_almost_equal, assert_allclose,
                           assert_array_equal, suppress_warnings)


class Tests:
    """
    Test object
    """
    def test_cases(self):
        # anchor
        bcs = {'x_max': 4,
               'y_max': 4,
               'edge_a': [5, 0, 3],  # 0-1 rigid load, 1-4 rigid
               'edge_b': 0,  # rigid
               'edge_c': 2,  # free
               'edge_d': 1}  # symmetry plane
        soil = {'cohesion': 0,
                'phi': 20,
                'unit_weight': 1}
        res, obj, lhs, rhs = DLO(Soil(soil), bcs).solve()
        # tas = scipy.io.loadmat('test_arrays.mat')
        # assert_array_equal(obj.toarray(), tas['obj'].toarray())
        # assert_array_equal(lhs.toarray(), tas['equalityMatrix'].toarray())
        # assert_array_equal(rhs.toarray(), tas['equalityRHS'].toarray())
        assert_almost_equal(res.fun, 6.951623304544706)

        # square
        bcs = {'x_max': 1,
               'y_max': 1,
               'edge_a': 0,  # rigid
               'edge_b': 2,  # free
               'edge_c': 3,  # flexible load
               'edge_d': 0}  # rigid
        soil = {'cohesion': 1,
                'phi': 0,  # no dilation, undrained soil or clay
                'unit_weight': 0}
        res = calc(bcs, soil, plot_mechanism=False)
        assert_almost_equal(res.fun, 1.999999999999789)

        # Prandtl
        bcs = {'x_max': 13,
               'y_max': 7,
               'edge_a': 0,  # rigid
               'edge_b': 0,  # rigid
               'edge_c': [5, 2, 9],  # 0-4 rigid load, 4-13 free
               'edge_d': 1}  # symmetry plane
        soil = {'cohesion': 1,
                'phi': 0,
                'unit_weight': 0}
        res = calc(bcs, soil, plot_mechanism=False)
        assert_almost_equal(res.fun, 5.205128205607322)

        # wall
        bcs = {'x_max': 9,
               'y_max': 6,
               'edge_a': 0,  # rigid
               'edge_b': 0,  # rigid
               'edge_c': 2,  # free
               'edge_d': 5}  # rigid load
        soil = {'cohesion': 1,
                'phi': 0,
                'unit_weight': 0}
        res = calc(bcs, soil, plot_mechanism=False)
        assert_almost_equal(res.fun, 2.602564102494697)


if __name__ == '__main__':
    Tests().test_cases()
