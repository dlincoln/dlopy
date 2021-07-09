from dlopy import dlo

# Anchor pullout: exact solution 6.912 (current unrefined numerical result below 6.951)

# |SYM
# |
# |----- \
# |SOIL| \
# |----- \
# |^\\\\\\
# |LOAD


bcs = {'x_max': 4,
       'y_max': 4,
       'edge_a': [5, 0, 3],  # 0-1 rigid load, 1-4 rigid
       'edge_b': 0,  # rigid
       'edge_c': 2,  # free
       'edge_d': 1}  # symmetry plane

soil = {'cohesion': 0,
        'phi': 20,
        'unit_weight': 1}

dlo.calc(bcs, soil, plot_mechanism=True)
