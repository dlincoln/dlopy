from dlopy import dlo

# Simple square: exact solution as numerically formulated 2.000

#    LOAD
#   VVVVVV
# \ ------
# \ |SOIL|
# \ ------
# \\\\\\\\

bcs = {'x_max': 1,
       'y_max': 1,
       'edge_a': 0,  # rigid
       'edge_b': 2,  # free
       'edge_c': 3,  # flexible load
       'edge_d': 0}  # rigid

soil = {'cohesion': 1,
        'phi': 0,  # no dilation, undrained soil or clay
        'unit_weight': 0}

dlo.calc(bcs, soil, plot_mechanism=True)
