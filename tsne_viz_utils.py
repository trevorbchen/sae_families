"""Shared KDE-density-contour plotting helpers, used by explore_reconstruction_tsne.py,
explore_feature_tsne.py, and explore_feature_lda.py instead of raw overplotted scatter --
at tens of thousands of points, scatter just becomes solid ink in the overlap regions.
Levels are relative to each class's own peak density (not a shared absolute scale) so a
diffuse class's thin tail doesn't dominate the view or blow up the axis range."""
import numpy as np
from scipy.stats import gaussian_kde


def _clamped_grid(xy_list, pad_frac=0.08, grid_n=150j, pct=(0.5, 99.5)):
    all_xy = np.concatenate(xy_list, axis=0)
    xmin, xmax = np.percentile(all_xy[:, 0], pct)
    ymin, ymax = np.percentile(all_xy[:, 1], pct)
    pad_x, pad_y = (xmax - xmin) * pad_frac, (ymax - ymin) * pad_frac
    xx, yy = np.mgrid[xmin - pad_x:xmax + pad_x:grid_n, ymin - pad_y:ymax + pad_y:grid_n]
    return xx, yy, (xmin - pad_x, xmax + pad_x), (ymin - pad_y, ymax + pad_y)


def kde_contours_multi(ax, xy_by_class, color_map, n_levels=4, max_samples=6000,
                        fill_alpha=0.07, line_alpha=0.85, seed=0):
    """One KDE contour set per class, all sharing one clamped grid/view so they're
    directly comparable. xy_by_class: dict label -> [N,2] array."""
    xx, yy, xlim, ylim = _clamped_grid(list(xy_by_class.values()))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    rng = np.random.default_rng(seed)
    for key, xy in xy_by_class.items():
        if xy.shape[0] < 5:
            continue
        sample = xy[rng.choice(xy.shape[0], size=min(xy.shape[0], max_samples), replace=False)]
        kde = gaussian_kde(sample.T)
        zz = kde(grid).reshape(xx.shape)
        levels = np.linspace(zz.max() * 0.12, zz.max(), n_levels)
        color = color_map.get(key, "#999999")
        ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.1, alpha=line_alpha)
        ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=fill_alpha, extend="neither")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def kde_contour_single(ax, xy, color, n_levels=5, max_samples=8000, fill_alpha=0.15,
                        line_alpha=0.9, seed=0, xlim=None, ylim=None):
    """Single-class KDE contour, e.g. the highlighted objective in a highlight plot."""
    if xlim is None or ylim is None:
        xx, yy, xlim2, ylim2 = _clamped_grid([xy])
        xlim = xlim or xlim2
        ylim = ylim or ylim2
        xx, yy = np.mgrid[xlim[0]:xlim[1]:150j, ylim[0]:ylim[1]:150j]
    else:
        xx, yy = np.mgrid[xlim[0]:xlim[1]:150j, ylim[0]:ylim[1]:150j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    rng = np.random.default_rng(seed)
    sample = xy[rng.choice(xy.shape[0], size=min(xy.shape[0], max_samples), replace=False)]
    kde = gaussian_kde(sample.T)
    zz = kde(grid).reshape(xx.shape)
    levels = np.linspace(zz.max() * 0.12, zz.max(), n_levels)
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.2, alpha=line_alpha)
    ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=fill_alpha, extend="neither")
    return xlim, ylim
