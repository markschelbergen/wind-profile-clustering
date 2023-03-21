def add_panel_labels(ax, offset_x=.15):
    import string
    if ax.ndim > 1:
        denominator = ax.shape[1]
        ax = ax.reshape(-1)
    else:
        denominator = ax.shape[0]
    for i, a in enumerate(ax):
        label = '('+string.ascii_lowercase[i]+')'
        if isinstance(offset_x, float):
            x = -offset_x
        else:
            x = -offset_x[i % denominator]
        if hasattr(a, 'text2D'):
            fun = a.text2D
        else:
            fun = a.text
        fun(x, .5, label, transform=a.transAxes, fontsize='large')  #, fontweight='bold', va='top', ha='right')