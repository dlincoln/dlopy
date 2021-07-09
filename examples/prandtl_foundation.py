from dlopy import dlo

# Symmetric strip foundation Prandtl mechanism: exact solution 2 + pi (current unrefined numerical result below 5.205)

# |LOAD
# |VVVV
# |------------- \
# |    SOIL    | \
# |------------- \
# |\\\\\\\\\\\\\\\
# |
# |SYM

bcs = {'x_max': 13,
       'y_max': 7,
       'edge_a': 0,  # rigid
       'edge_b': 0,  # rigid
       'edge_c': [5, 2, 9],  # 0-4 rigid load, 4-13 free
       'edge_d': 1}  # symmetry plane

soil = {'cohesion': 1,
        'phi': 0,
        'unit_weight': 1}

dlo.calc(bcs, soil, plot_mechanism=True)
