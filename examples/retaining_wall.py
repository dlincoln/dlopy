from dlopy import dlo

# Retaining wall: exact solution (2 + pi) / 2 (current unrefined numerical result below 2.603)

# L > ---------- \
# O > |  SOIL  | \
# A > |        | \
# D > ---------- \
#     \\\\\\\\\\\\

bcs = {'x_max': 9,
       'y_max': 6,
       'edge_a': 0,  # rigid
       'edge_b': 0,  # rigid
       'edge_c': 2,  # free
       'edge_d': 5}  # rigid load

soil = {'cohesion': 1,
        'phi': 0,
        'unit_weight': 0}

dlo.calc(bcs, soil, plot_mechanism=True)
