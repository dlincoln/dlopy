from numpy import vstack, column_stack, array, ndarray, zeros, ones, mgrid, insert
from math import gcd, tan, pi
from scipy import sparse, optimize
from matplotlib import pyplot, collections
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import triu
from numba import guvectorize


class Soil(object):
    """
    Soil object
    """
    def __init__(self, soil):
        _id = None
        _type = 'single soil'
        self.cohesion = soil['cohesion']
        self.phi = soil['phi']  # degrees
        self.unit_weight = soil['unit_weight']


class DLO(object):
    """
    DLO object for rectangular soil domain
    Efficient Python Implementation of Discontinuity Layout Optimization (DLO) Demonstration
    (C) 2009 Computational Mechanics & Design Research Group, University of Sheffield
    https://cmd.shef.ac.uk/?q=publications/application-discontinuity-layout-optimization-geotechnical-limit-analysis-problems

    Boundary conditions
    edges: . c .  types: 0 = rigid,        1 = symmetry plane
           d   b         2 = free          3 = 'flexible' load
           . a .         4 = 'rigid' load  5 = 'rigid' load constrained laterally

    If two conditions along edge_a/c replace with list: [edge_a/c1_type, edge_a/c2_type, length_of_edge_a/c2]

    References
    M. Gilbert et al. Application of discontinuity layout optimization to geotechnical limit analysis problems
    Proceedings of the 7th European Conference on Numerical Methods in Geotechnical Engineering, 2010.
    """
    def __init__(self, soil, bcs):
        self.s = soil
        self.x = bcs['x_max']
        self.y = bcs['y_max']
        self.xym = array((bcs['x_max'], bcs['y_max']))
        self.bcs = zeros((4, 3), dtype=int)
        self.bcs[0, :] = bcs['edge_a'] if isinstance(bcs['edge_a'], list) else [bcs['edge_a'], 0, 0]
        self.bcs[1, :] = [bcs['edge_b'], 0, 0]
        self.bcs[2, :] = bcs['edge_c'] if isinstance(bcs['edge_c'], list) else [bcs['edge_c'], 0, 0]
        self.bcs[3, :] = [bcs['edge_d'], 0, 0]
        self.nodes, self.nn = self.nodes()
        self.discs, self.lens, self.nd = self.discontinuities()
        self.b = self.compatibility_matrix()
        self.obj_p, self.pad_n, self.n = self.plastic_multiplier_terms()
        self.f_d = self.self_weight()
        self.f_l = self.unit_load()
        self.tied = self.tie_discs()
        self.result = None

    def nodes(self):
        """
        Create nodes
        :return: nodes
        """
        x, y = mgrid[0:self.x+1:1, 0:self.y+1:1]
        n = vstack([x.ravel(), y.ravel()]).T

        return n, len(n)

    @staticmethod
    @guvectorize(['f8[:, :], i4[:, :], f8[:], i4[:]'], '(n,m),(a,b),(c)->()')
    def type(x, e, m, i):
        x1, y1, x2, y2 = x[0, 0], x[0, 1], x[1, 0], x[1, 1]
        a, b, c, d, xm, ym = e[0], e[1], e[2], e[3], m[0], m[1]
        a1 = a[0] * ((y1 + y2 == 0) and (x1 <= (xm - a[2])) and (x2 <= (xm - a[2])))
        a2 = a[1] * ((y1 + y2 == 0) and (x1 >= (xm - a[2])) and (x2 >= (xm - a[2])))
        b0 = b[0] * (x1 + x2 == 2 * xm)
        c1 = c[0] * ((y1 + y2 == 2 * ym) and (x1 <= (xm - c[2])) and (x2 <= (xm - c[2])))
        c2 = c[1] * ((y1 + y2 == 2 * ym) and (x1 >= (xm - c[2])) and (x2 >= (xm - c[2])))
        d0 = d[0] * (x1 + x2 == 0)
        i[:] = a1 + a2 + b0 + c1 + c2 + d0

    @staticmethod
    @guvectorize(['f8[:, :], b1[:]'], '(n,m)->()')
    def gcd(x, b):
        b[:] = gcd(abs(x[0, 0] - x[1, 0]), abs(x[0, 1] - x[1, 1])) < 1.0001

    def discontinuities(self):
        """
        Create discontinuities
        :return: discs
        """
        n = self.nodes
        adj = triu(kneighbors_graph(n, self.nn-1, mode='distance', n_jobs=-1), k=0, format='coo')
        ind = column_stack((adj.row, adj.col))
        lns = column_stack(adj.data).T
        dtk = self.gcd(n[ind])
        ind = ind[dtk]
        bcs = self.type(n[ind], self.bcs, self.xym)
        lns = lns[dtk].ravel()
        dis = column_stack((ind+1, bcs, list(range(len(bcs)))))

        return dis, lns, len(dis)

    @staticmethod
    @guvectorize(['i4[:], f8, f8[:, :], f8[:], i4[:], i4[:], f8[:]'], '(n),(),(a,b),(c)->(c),(c),(c)')
    def cm(d, ln, n, s, r, c, b):
        n1, n2, i = d[0], d[1], d[3]
        c1, c2, v = 2*n1-1, 2*n2-1, 2*(i+1)-1
        cos, sin = (n[n1-1, 0] - n[n2-1, 0]) / ln, (n[n1-1, 1] - n[n2-1, 1]) / ln
        r[:] = [c1-1, c1-1, c1, c1, c2-1, c2-1, c2, c2]
        c[:] = [v-1, v, v-1, v, v-1, v, v-1, v]
        b[:] = [cos, -sin, sin, cos, -cos, sin, -sin, -cos]

    def compatibility_matrix(self):
        """
        Setup compatibility matrix
        :return: b
        """
        r, c, d = self.cm(self.discs, self.lens, self.nodes, ndarray((8,)))
        bb = sparse.coo_matrix((d.ravel(), (r.ravel(), c.ravel())))
        return bb

    @staticmethod
    @guvectorize(['i4[:], f8, i4, f8[:], f8[:], f8[:], i4[:], i4[:], f8[:], i4[:], i4[:], i4[:], f8[:]'],
                 '(n),(),(),(p),(s1),(s2)->(s1),(s1),(s1),(s2),(s2),(s2),(s2)')
    def pmt(d, ln, ct, p, s1, s2, r1, c1, n, r2, c2, pn, op):
        eff_c, eff_tp = p[0] * (d[2] != 1), p[1] * (d[2] != 1)
        cv = 2*ct  # 2 * count - 1, 2 * count - 1
        r1[:] = [cv, cv, cv+1, cv+1]
        c1[:] = [cv, cv + 1, cv, cv + 1]
        n[:] = [1, -1, eff_tp, eff_tp]
        r2[:] = [cv, cv+1]
        c2[:] = [2*d[3], 2*d[3]+1]
        pn[:] = [-1, -1]
        op[:] = [eff_c*ln, eff_c*ln]

    def plastic_multiplier_terms(self):
        """
        Setup plastic multiplier terms
        :return: obj_p, pad_n, n
        """
        par = [self.s.cohesion, tan(pi * self.s.phi / 180)]
        msk = self.discs[:, 2] < 2
        pm = self.pmt(self.discs[msk], self.lens[msk], range(msk.sum()), par, ndarray((4,)), ndarray((2,)))
        nn = sparse.coo_matrix((pm[2].ravel(), (pm[0].ravel(), pm[1].ravel())))
        pn = sparse.coo_matrix((pm[5].ravel(), (pm[3].ravel(), pm[4].ravel())), shape=(2 * msk.sum(), 2 * self.nd))
        op = sparse.coo_matrix((pm[6].ravel(), (pm[3].ravel(), pm[4].ravel()*0)))

        return op, pn, nn

    @staticmethod
    @guvectorize(['i4[:], f8, f8[:, :], f8, f8, f8[:], i4[:], f8[:]'], '(n),(),(a,b),(),(),(s)->(s),(s)')
    def sw(d, ln, n, uw, ym, s, r, f):
        n1, n2, i = d[0], d[1], d[3]
        cos, sin = (n[n1 - 1, 0] - n[n2 - 1, 0]) / ln, (n[n1 - 1, 1] - n[n2 - 1, 1]) / ln
        wdt = n[n2 - 1, 0] - n[n1 - 1, 0]
        wgt = uw * wdt * (ym - 0.5 * (n[n1 - 1, 1] + n[n2 - 1, 1]))
        r[:] = [2 * i, 2 * i + 1]
        f[:] = [-sin * wgt, -cos * wgt]

    def self_weight(self):
        """
        Self weight loads
        :return: f_d
        """
        fd = self.sw(self.discs, self.lens, self.nodes, self.s.unit_weight, self.xym[1], ndarray((2,)))
        fd = sparse.coo_matrix((fd[1].ravel(), (fd[0].ravel(), fd[0].ravel()*0)))

        return fd

    @staticmethod
    @guvectorize(['i4[:], i4[:], i4[:]'], '(n)->(),()')
    def ul(d, r, f):
        i = d[3]
        r[:] = 2*(i+1)-1
        f[:] = 1

    def unit_load(self):
        """
        External live loads
        :return: f_l
        """
        msk = self.discs[:, 2] >= 3
        fl = self.ul(self.discs[msk])
        fl = sparse.coo_matrix((fl[1], (fl[0]*0, fl[0])), shape=(1, 2*self.nd))

        return fl

    @staticmethod
    @guvectorize(['i4[:], i4, i4, f8[:], i4[:], i4[:], i4[:]'], '(n),(),(),(s)->(s),(s),(s)')
    def td(d, il, ct, s, r, c, e):
        bc, i = d[0], d[1]
        r[:] = [2 * ct, 2 * ct, 2 * ct + 1, 2 * ct + 1]
        if (il != -1) or (bc == 5):
            if il == -1:
                e[:] = [1, 0, 0, 0]
                c[:] = [2 * (i + 1) - 2, 0, 0, 0]
            else:
                c[:] = [2 * (i + 1) - 2, 2 * (il + 1) - 2, 2 * (i + 1) - 1, 2 * (il + 1) - 1]
                e[:] = [1, -1, 1, -1]

    def tie_discs(self):
        """
        Link element displacements
        :return: tied
        """
        msk = self.discs[:, 2] >= 4
        if any(msk):
            td = self.td(self.discs[msk, 2:], insert(self.discs[msk, 3], 0, -1)[:-1], range(msk.sum()), ndarray((4,)))
            td = sparse.coo_matrix((td[2].ravel(), (td[0].ravel(), td[1].ravel())), shape=(2 * msk.sum(), 2 * self.nd))
        else:
            td = sparse.coo_matrix((0, 2 * self.nd))

        return td

    def solve(self):
        """
        Cast and solve linear programming problem
        :return: res
        """
        bn = 1000
        pad_b = sparse.coo_matrix((self.b.shape[0], self.n.shape[1]))
        pad_f_l = sparse.coo_matrix((1, self.n.shape[1]))
        pad_tied = sparse.coo_matrix((self.tied.shape[0], self.n.shape[1]))
        a_eq = sparse.vstack([sparse.hstack([self.b, pad_b]), sparse.hstack([self.pad_n, self.n]),
                              sparse.hstack([self.f_l, pad_f_l]), sparse.hstack([self.tied, pad_tied])])

        b_eq = sparse.lil_matrix((a_eq.shape[0], 1))
        b_eq[self.b.shape[0] + self.n.shape[0], 0] = 1

        c = sparse.vstack([self.f_d, self.obj_p])

        l_b = sparse.vstack([sparse.coo_matrix(-bn * ones((self.b.shape[1], 1))),
                             sparse.coo_matrix((self.n.shape[1], 1))])
        u_b = sparse.vstack([sparse.coo_matrix(bn * ones((self.b.shape[1], 1))),
                             sparse.coo_matrix(bn * ones((self.n.shape[1], 1)))])
        bounds = sparse.hstack([l_b, u_b])

        res = optimize.linprog(c.toarray(),
                               A_ub=None,
                               b_ub=None,
                               A_eq=a_eq,
                               b_eq=b_eq.toarray(),
                               bounds=bounds.toarray(),
                               method='highs',
                               options={'disp': True})
        # note solver methods 'interior-point' (default) 'simplex' (legacy) perform poorly for these lp problems
        # todo test sparse solve
        self.result = res

        return res, c, a_eq, b_eq

    def plot_mechanism(self):
        """
        Plot mechanism
        :return:
        """
        active = abs(self.result.x[0:2 * self.nd]).reshape(self.nd, 2).sum(axis=1) > 0.001
        lines_i = list(zip(self.nodes[self.discs[:, 0] - 1],
                           self.nodes[self.discs[:, 1] - 1]))
        lines_a = list(zip(self.nodes[self.discs[active, 0] - 1],
                           self.nodes[self.discs[active, 1] - 1]))
        lc_i = collections.LineCollection(lines_i, colors='grey', linewidths=0.5)
        lc_a = collections.LineCollection(lines_a, colors='red', linewidths=1.5)
        fig, ax = pyplot.subplots()
        ax.add_collection(lc_i)
        ax.add_collection(lc_a)
        ax.axis('equal')
        # ax.autoscale()
        # ax.margins(0.1)
        pyplot.xticks(range(0, self.x+1, 1))
        pyplot.yticks(range(0, self.y+1, 1))
        pyplot.show()


def calc(bcs, soil, plot_mechanism=True):
    """
    Calculation type for simple single rectangular soil
    :param bcs: [dict] boundary conditions
    :param soil: [dict] soil properties
    :param plot_mechanism: [boolean] plot mechanism
    :return: [object] optimisation results
    """
    # create soil type
    s = Soil(soil)

    # create dlo calculation with single rectangular soil type
    print('Creating arrays...')
    c = DLO(s, bcs)

    # solve
    print('Solving...')
    result = c.solve()[0]
    print(result)

    # plot
    if plot_mechanism:
        c.plot_mechanism()
    return result


if __name__ == "__main__":
    pass
