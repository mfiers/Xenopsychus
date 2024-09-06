"""Xenopsychus - hexbins for scanpy


chatgpt thought Xenopsychus was the latin name for the angler fish -
it does not seem to be - but who cares...

"""

import logging
import math
from copy import copy
from dataclasses import dataclass
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from scipy.stats import binomtest
# from statsmodels.stats.multitest import multipletests

lg = logging.getLogger(__name__)
lg.setLevel(logging.INFO)

MINCNT_FAIL = "MINCNT@!" * 4
SUBSET_FAIL = "SUBSETT@!" * 4
CAT_FAIL = "CATCAT@!" * 4

#
# Decorators!
#


def mincnt(method):
    """
    Filter the hexbin to remove all cells with < self.mincnt
    spots in there.
    """

    def decorator(self, *args, **kwargs):
        mincnt = kwargs.pop("mincnt", self.mincnt)
        mincnt_fail_color = kwargs.pop(
            "mincnt_fail_color", self.mincnt_fail_color
        )

        method(self, *args, **kwargs)
        if self.mincnt > 0:
            # need to draw the thing first - to calculate colors
            self.ax.figure.canvas.draw()
            facecolors = self.hb.get_facecolors()
            cellcnt = (
                self.data_subset["_hb"]
                .value_counts()
                .reindex(self.data["_hb"].unique())
                .fillna(0)
                .sort_index()
            )
            for i, cnt in enumerate(cellcnt):
                if cnt <= mincnt:
                    facecolors[i] = self.ensure_rgba(mincnt_fail_color)
            self.hb.set(array=None, facecolors=facecolors)

    return decorator


def add_colorbar(method):
    def decorator(self, *args, **kwargs):
        colorbar = kwargs.pop("colorbar", self.colorbar)

        rv = method(self, *args, **kwargs)
        if colorbar:
            ax = self.ax
            divider = make_axes_locatable(ax)
            self.cax = divider.append_axes("right", size="5%", pad=0.05)
            self.fig.colorbar(self.hb, cax=self.cax, orientation="vertical")

    return decorator


def addborder(method):
    def decorator(self, *args, **kwargs):
        border_x = kwargs.pop("border_x", self.border_x)
        border_y = kwargs.pop("border_y", self.border_y)
        method(self, *args, **kwargs)
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        xd = border_x * (xmax - xmin)
        yd = border_y * (ymax - ymin)
        self.ax.set_xlim(xmin - xd, xmax + xd)
        self.ax.set_ylim(ymin - yd, ymax + yd)

    return decorator


def ensure_ax(method):
    def decorator(self, *args, **kwargs):
        ax = None
        if "ax" in kwargs:
            ax = kwargs["ax"]
            del kwargs["ax"]

        if ax is None:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.ax = self.fig.gca()
        else:
            self.ax = ax
            self.fig = self.ax.figure

        method(self, *args, **kwargs)

    return decorator


def clean_spines(method):
    def decorator(self, *args, **kwargs):
        showaxes = self.getarg(kwargs, "showaxes", False)

        method(self, *args, **kwargs)

        ax = self.ax
        if not showaxes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xlabel(self.x_name)
            ax.set_ylabel(self.y_name)

        ax.grid(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    return decorator


def subset(method):
    def decorator(self, *args, **kwargs):
        subset = kwargs.pop("subset", self.subset)
        if subset is None:
            self.data_subset = self.data
        else:
            self.data_subset = self.data[subset]
        method(self, *args, **kwargs)

    return decorator


def nosubset(method):
    def decorator(self, *args, **kwargs):
        subset = kwargs.pop("subset", self.subset)
        self.data_subset = self.data
        if subset is not None:
            print("Warning - cannot subset on plot_agg!")
        method(self, *args, **kwargs)

    return decorator


def cluster_marker_lines(method):
    def decorator(self, *args, **kwargs):
        cluster_show = kwargs.pop("cluster_show", self.cluster_show)
        cluster_linecolor = kwargs.pop(
            "cluster_linecolor", self.cluster_linecolor
        )
        cluster_outline = kwargs.pop("cluster_outline", self.cluster_outline)
        cluster_linewidth = kwargs.pop(
            "cluster_linewidth", self.cluster_linewidth
        )

        method(self, *args, **kwargs)

        def find_borders(row, q, direction):
            dx, dy = dict(
                left=(0, -2),
                topleft=(1, -1),
                topright=(1, 1),
                right=(0, 2),
                bottomleft=(-1, -1),
                bottomright=(-1, 1),
            )[direction]

            x = row["r1"] + dx
            y = row["r0"] + dy
            oc = q[(q.r1 == x) & (q.r0 == y)]
            if len(oc) == 0:
                return 1
            elif oc.iloc[0]["array"] == row["array"]:
                return 0
            else:
                return 2

        if cluster_show is not None:
            cats = self.find_categories(self.data, cluster_show)
            valldata = pd.DataFrame(self.hb.get_offsets())
            valldata["array"] = list(cats)
            valldata["r0"] = valldata[0].rank(method="dense").astype(int)
            valldata["r1"] = valldata[1].rank(method="dense").astype(int)

            vertices = self.hb.get_paths()[0].vertices
            dirsel = [
                ("right", slice(0, 2)),
                ("topright", slice(1, 3)),
                ("topleft", slice(2, 4)),
                ("left", slice(3, 5)),
                ("bottomleft", slice(4, 6)),
                ("bottomright", slice(5, 7)),
            ]

            rcutoff = 1 if cluster_outline else 2
            for direction, vsel in dirsel:
                borders = valldata.apply(
                    find_borders, q=valldata, direction=direction, axis=1
                )
                for i, (_, r) in enumerate(borders.items()):
                    if r >= rcutoff:
                        xx = vertices[vsel, 0] + valldata.iloc[i][0]
                        yy = vertices[vsel, 1] + valldata.iloc[i][1]
                        self.ax.plot(
                            xx,
                            yy,
                            c=cluster_linecolor,
                            zorder=10,
                            lw=cluster_linewidth,
                        )

    return decorator


def supadec(method):
    """
    Combine a set of common decorators

    Args:
        method: a method of this object

    Returns:
        method: decorated methods
    """
    return cluster_marker_lines(
        mincnt(addborder(clean_spines(ensure_ax(method))))
    )


class Xenopsychus:

    def __init__(
        self,
        data,
        x_name,
        y_name,
        figsize=(5, 4),
        dpi=120,
        colorbar=True,
        cmap_cat="Set2",
        mincnt=5,
        palette=None,
        border_x=0.0,
        border_y=0.0,
        cluster_show=None,
        cluster_linecolor="black",
        cluster_outline=False,
        cluster_linewidth=1.5,
        cat_fail_color="lightgrey",
        mincnt_fail_color="lightgrey",
        subset_fail_color="lightgrey",
        subset=None,
        **plotargs
    ):

        # core -
        self.data = data
        self.y_name = y_name
        self.x_name = x_name

        # plotting
        self.figsize = figsize
        self.dpi = dpi

        # draw cluster lines
        self.cluster_show = cluster_show
        self.cluster_linecolor = cluster_linecolor
        self.cluster_outline = cluster_outline
        self.cluster_linewidth = cluster_linewidth

        # categorical colors
        self.cmap_cat = cmap_cat
        self.palette = palette
        self.cat_fail_color = cat_fail_color

        self.border_x = border_x
        self.border_y = border_y

        # filter min counts
        self.mincnt = mincnt
        self.mincnt_fail_color = mincnt_fail_color

        # arguments for hexbin
        self.plotargs = plotargs

        # add a colorbar?
        self.colorbar = colorbar

        # subsetting data
        # needs to be done carefully - we take the bin ids' from the full dataset.
        # and then calculate a second 'data for the aggregation & counting
        self.subset = subset
        self.subset_fail_color = subset_fail_color

        # set a few defaults for hexbin
        # can be overridden by anything in the
        # __init___ call.
        for k, v in dict(
            mincnt=1,
            linewidths=0.25,
            gridsize=16,
            edgecolors="black",
            cmap="YlGnBu",
        ).items():
            self.plotargs[k] = self.plotargs.get(k, v)

        # pull x, y vectors from self.data
        self.plotargs["x"] = self.data[self.x_name]
        self.plotargs["y"] = self.data[self.y_name]

        # first step - calculate hexin IDs for every cell
        self.data["_hb"] = binbin(
            x=self.plotargs["x"],
            y=self.plotargs["y"],
            gridsize=self.plotargs["gridsize"],
        )

    #
    # Helper functions
    #

    @classmethod
    def ensure_rgba(cls, x):
        if isinstance(x, str):
            return mpl.colors.to_rgba(x)
        x = tuple(x)
        if len(x) == 3:
            return (x[0], x[1], x[2], 1)
        else:
            return x

    def apply_agg(self, agg, fillna=0, **kwargs):
        agg = (
            agg.reindex(self.data["_hb"].unique()).fillna(fillna).sort_index()
        )
        self.agg = agg
        self.hb.set(array=agg.values)
        if kwargs.get("vmin") is None:
            kwargs["vmin"] = agg.quantile(0.025)
        if kwargs.get("vmax") is None:
            kwargs["vmax"] = agg.quantile(0.975)
        self.hb.set_norm(
            mpl.colors.Normalize(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
        )

    def getarg(self, kwargs, key, default):
        rv = None
        found_rv = False

        if key in self.plotargs:
            rv = self.plotargs[key]
            found_rv = True
            del self.plotargs[key]

        if key in kwargs:
            rv = kwargs[key]
            found_rv = True
            del kwargs[key]

        if found_rv:
            return rv
        else:
            return default

    def find_categories(self, data, C, fracdiff=0.1):

        def count_most(r):
            """
            Return the most abundant category

            Args:
                r (pd.Series): observations in this cell

            Returns:
                ANY: most abundant value
            """
            vc = r.value_counts()
            mx = vc.iloc[0]
            sum = vc.sum()
            ml = vc.index[0]
            if len(vc) == 1:
                return ml
            else:
                m2 = vc.iloc[1]
                if ((mx - m2) / sum) <= fracdiff:
                    return CAT_FAIL
                else:
                    return ml

        agg = data.groupby("_hb")[C].agg(count_most)
        agg = (
            agg.reindex(self.data["_hb"].unique())
            .fillna(SUBSET_FAIL)
            .sort_index()
        )
        return agg

    #
    # Plotting functions
    #

    @add_colorbar
    @supadec
    @nosubset
    def plot_agg(self, agg, **kwargs):

        pa = copy(self.plotargs)
        pa.update(kwargs)

        # dummy plot
        self.hb = self.ax.hexbin(**pa)
        agg = agg.sort_index()
        self.hb.set(array=agg.values)
        if kwargs.get("vmin") is None:
            kwargs["vmin"] = agg.quantile(0.025)
        if kwargs.get("vmax") is None:
            kwargs["vmax"] = agg.quantile(0.975)
        self.hb.set_norm(
            mpl.colors.Normalize(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
        )

    @add_colorbar
    @supadec
    @subset
    def plot_num(self, C, ax=None, **kwargs):
        pa = copy(self.plotargs)
        pa.update(kwargs)
        self.hb = self.ax.hexbin(**pa)

        agg = self.data_subset.groupby("_hb")[C].mean()
        self.apply_agg(agg, **pa)

    # @add_colorbar
    # @supadec
    # @subset
    # def plot_num_diff(self, C, D, ax=None, **kwargs):
    #     pa = copy(self.plotargs)
    #     pa.update(kwargs)
    #     self.hb = self.ax.hexbin(**pa)
    #     vext = 3

    #     def numdiff(r, val, group):
    #         A = r[r[group]][val]
    #         B = r[~r[group]][val]
    #         if min(len(A), len(B)) < 5:
    #             return np.nan
    #         elif B.mean() == 0:
    #             return vext
    #         rv = np.log2(A.mean() / B.mean())
    #         if rv < -vext:
    #             return -vext
    #         elif rv > vext:
    #             return vext
    #         else:
    #             return rv

    #     agg = self.data_subset.groupby("_hb")[[C, D]].apply(
    #         numdiff, val=C, group=D
    #     )
    #     if kwargs.get("vmin") is None:
    #         kwargs["vmin"] = agg.quantile(0.025)
    #     if kwargs.get("vmax") is None:
    #         kwargs["vmax"] = agg.quantile(0.975)

    #     vmin, vmax = kwargs["vmin"], kwargs["vmax"]
    #     vext = min(vext, max(vmin, vmax))
    #     pa["vmin"], pa["vmax"] = -vext, vext
    #     pa["cmap"] = "coolwarm"
    #     self.apply_agg(agg, **pa)

    @supadec
    @subset
    def plot_cat(self, C, palette=None, fracdiff=0.1, **kwargs):
        """
        Plot categorical hexbin

        Args:
            C (str): Column in .data to plot
            palette (dict): field to color map
            fracdiff (float): min fraction difference
            mincnt (int): min no points to assign
            failcol (str or (r,g,b)): color in case of unclear assignment

        Returns:
            hexbin
        """

        subset_fail_color = kwargs.pop(
            "subset_fail_color", self.subset_fail_color
        )
        cat_fail_color = kwargs.pop("cat_fail_color", self.cat_fail_color)
        mincnt_fail_color = kwargs.pop(
            "mincnt_fail_color", self.mincnt_fail_color
        )

        pa = copy(self.plotargs)
        pa.update(kwargs)

        if palette is None:
            if self.palette is None:
                allcats = sorted(self.data[C].unique())
                cmap = plt.colormaps.get(self.cmap_cat)
                palette = {x: cmap(i) for (i, x) in enumerate(allcats)}
            else:
                palette = self.palette

        # ensure we can fail - add cat_fail_color
        palette[CAT_FAIL] = cat_fail_color
        palette[MINCNT_FAIL] = mincnt_fail_color
        palette[SUBSET_FAIL] = subset_fail_color

        self.hb = self.ax.hexbin(**pa)
        agg = self.find_categories(
            C=C, data=self.data_subset, fracdiff=fracdiff
        )
        facecolors = [palette[x] for x in agg.sort_index().values]
        self.hb.set(array=None, facecolors=facecolors)

    @add_colorbar
    @supadec
    @subset
    def plot_count(self, **kwargs):
        # do a regular one to have a hexbin
        pa = copy(self.plotargs)
        pa.update(kwargs)
        self.hb = self.ax.hexbin(**pa)
        # calculate based on data_subset
        agg = self.data_subset["_hb"].value_counts()
        self.apply_agg(agg, fillna=0, **pa)

    @add_colorbar
    @supadec
    @subset
    def plot_count_diff(self, C, vext=3, method="delta", **kwargs):

        # do a regular one to have a hexbin
        pa = copy(self.plotargs)
        pa.update(kwargs)
        self.hb = self.ax.hexbin(**pa)

        from scipy.stats import binom

        def binom_diff(r, val, dp):
            # print(r[val].head())
            k = (r[val]).sum()
            n = len(r)
            p = k / n
            if p < dp:
                rv = np.log10(binom.cdf(k, n, dp))
            else:
                rv = -np.log10(binom.sf(k, n, dp))
            # print(k, n, p, dp, rv)
            return rv

        def numdiff(r, val):
            A = r[val].sum()
            B = len(r) - A
            if A + B < 2:
                return np.nan
            return B - A

        if method == "delta":
            aggfunc = numdiff
        elif method == "binomlp":
            p = sum(self.data_subset[C]) / len(self.data_subset[C])
            # print(C)
            # print(self.data_subset[C].value_counts())
            # print(p)
            aggfunc = partial(binom_diff, dp=p)

        agg = self.data_subset.groupby("_hb")[[C]].apply(aggfunc, val=C)

        vext = max(vext, max(abs(agg.quantile(0.1)), abs(agg.quantile(0.9))))

        if "vmin" in kwargs:
            pa["vmin"] = kwargs["vmin"]
        else:
            pa["vmin"] = -vext

        if "vmax" in kwargs:
            pa["vmax"] = kwargs["vmax"]
        else:
            pa["vmax"] = vext

        self.vext = vext
        self.apply_agg(agg, fillna=0, **pa)


def binbin(x, y, gridsize, **xargs):
    """
    Calculate bin ids for hexbin assignment.

    Does not work with logscale!

    Code graciously borrowed from the matplotlib library
    - because tbh - I do not quite understand how they do it...

    """

    # Set the size of the hexagon grid
    if np.iterable(gridsize):
        nx, ny = gridsize
    else:
        nx = gridsize
        ny = int(nx / math.sqrt(3))

    # Count the number of data in each hexagon
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # remnant from mpl code - for possible log transform
    tx, ty = x, y

    xmin, xmax = (tx.min(), tx.max()) if len(x) else (0, 1)
    ymin, ymax = (ty.min(), ty.max()) if len(y) else (0, 1)

    # to avoid issues with singular data, expand the min/max pairs
    xmin, xmax = mtransforms.nonsingular(xmin, xmax, expander=0.1)
    ymin, ymax = mtransforms.nonsingular(ymin, ymax, expander=0.1)

    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny

    # In the x-direction, the hexagons exactly cover the region from
    # xmin to xmax. Need some padding to avoid roundoff errors.
    padding = 1.0e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny

    # Positions in hexagon index coordinates.
    ix = (tx - xmin) / sx
    iy = (ty - ymin) / sy

    ix1 = np.round(ix).astype(int)
    iy1 = np.round(iy).astype(int)
    ix2 = np.floor(ix).astype(int)
    iy2 = np.floor(iy).astype(int)

    # flat indices, plus one so that out-of-range points go to position 0.
    i1 = np.where(
        (0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1),
        ix1 * ny1 + iy1 + 1,
        0,
    )
    i2 = np.where(
        (0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2),
        ix2 * ny2 + iy2 + 1,
        0,
    )

    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = d1 < d2

    # here I take over again
    # calculate per orignal data point the
    # final hexbin ID.
    idb1 = pd.Series(i1) - 1
    idb2 = pd.Series(i2) + (nx1 * ny1 - 1)
    idb1.loc[~bdist] = idb2.loc[~bdist]
    return list(idb1)


# def get_array_diffscore(C, diff_a, diff_b,
#                         agg_func, subset=None, **hbargs):
#     """
#     Prepare the array for diff-score calculation
#     Also, execute the agg_func &
#     """

#     diff_a_original = diff_a.copy()
#     if subset is not None:
#         diff_a = diff_a & subset


#     if diff_b is None:
#         diff_b = (~diff_a_original) & subset

#     bins = binbin(**hbargs)

#     D = pd.DataFrame(dict(
#         a=diff_a, b=diff_b, v=C,
#         bin=list(bins)))


#     arr = agg_func(D).sort_index()
#     return D, arr


# def get_array_score(C, agg_func, subset=None,
#                     **hbargs):
#     """
#     Prepare the array for score calculation
#     And execute the agg_func
#     """
#     bins = binbin(**hbargs)

#     D = pd.DataFrame(dict(
#         v=C, bin=bins))

#     if subset is None:
#         D['subset'] = True
#     else:
#         D['subset'] = subset

#     agg = agg_func(D)

#     #print(agg.head())
#     #print("D", len(D), "agg", len(agg))

#     #print('cc')
#     #print(D.head())
#     #print(arr)

#     return D, agg


# def get_hexbin_categorical(ax, C, generate_OR=False, **hbargs):

#     def _most_abundant(values):
#         # return most abundant value for when we'er in category modus

#         # find frequency of each value
#         vc = pd.Series(values).value_counts()
#         if len(vc) == 0:
#             return None
#         return vc.index[0]


#     def _OR(values):
#         vc = pd.Series(values).value_counts()
#         if len(vc) == 1:
#             return 100
#         elif len(vc) > 1:
#             return vc.iloc[0] / vc.iloc[1]
#         else:
#             return 1

#     try:
#         C = C.astype(int).astype('category')
#     except:
#         C = C.cat.codes

#     rf = _OR if generate_OR else _most_abundant
#     import streamlit as st

#     #st.write(hbargs)
#     #print(C)
#     return ax.hexbin(C=C, reduce_C_function=rf, **hbargs)


# def agg_generic(D, aggfunc="mean"):

#     #filter out subset
#     DD = D.copy()
#     DD.loc[~DD['subset'], 'v'] = np.nan
#     agg = pd.DataFrame(dict(
#         score = DD.groupby('bin')['v'].agg(aggfunc),
#         count = DD.groupby('bin')['v'].count()
#         ))
#     agg['score'] = agg['score'].fillna(agg['score'].min())

#     agg = agg.sort_index()
#     return agg


# def agg_diff_lfc(D, norm=False, aggfunc=np.mean):
#     "Take LFC of normalized means per bin."

#     if norm:
#         D.loc[D.a, 'v'] = D.loc[D.a, 'v'] / D.loc[D.a, 'v'].mean()
#         D.loc[D.b, 'v'] = D.loc[D.b, 'v'] / D.loc[D.b, 'v'].mean()

#     agg = pd.DataFrame(dict(
#         a = D[D['a']].groupby('bin')['v'].agg(aggfunc),
#         b = D[D['b']].groupby('bin')['v'].agg(aggfunc),
#         cnt_a = D[D['a']].groupby('bin')['v'].count(),
#         cnt_b = D[D['b']].groupby('bin')['v'].count(),
#     ))


#     agg['cnt_a'] = agg['cnt_a'].fillna(0).astype(int)
#     agg['cnt_b'] = agg['cnt_b'].fillna(0).astype(int)
#     agg['count'] = np.minimum(agg['cnt_a'], agg['cnt_b'])

#     agg['score'] = (np.log2(agg['a'] / agg['b']))\
#         .replace([-np.inf, np.inf], 0)\
#         .fillna(0)

#     #print(agg)

#     return agg


# def agg_diff_delta(D, norm=False, aggfunc=np.mean):
#     "Take LFC of normalized means per bin."

#     from scipy.stats import mannwhitneyu
#     from statsmodels.stats.multitest import multipletests


#     if norm:
#         D.loc[D.a, 'v'] = D.loc[D.a, 'v'] / D.loc[D.a, 'v'].mean()
#         D.loc[D.b, 'v'] = D.loc[D.b, 'v'] / D.loc[D.b, 'v'].mean()

#     D['v1'] = D['v'].copy()
#     D['v2'] = D['v'].copy()
#     D.loc[~D['a'], 'v1'] = np.nan
#     D.loc[~D['b'], 'v2'] = np.nan
#     D['n1'] = ~D['v1'].isna()
#     D['n2'] = ~D['v2'].isna()


#     mwu_bin = {}

#     for binname, group in D.groupby('bin'):
#         n1, n2 = group['n1'].sum(), group['n2'].sum()
#         if min(n1, n2) < 5:
#             mwu_bin[binname] = dict(pvalue=1, stat=0, a=0, b=0. )
#             continue
#         v1, v2 = group['v1'].dropna(), group['v2'].dropna()
#         x = mannwhitneyu(v1, v2)
#         mwu_bin[binname] = dict(
#             pvalue=x.pvalue, stat=x.statistic,
#             a = aggfunc(v1), b=aggfunc(v2),
#             cnt_a = n1, cnt_b = n2,
#         )

#     agg = pd.DataFrame(mwu_bin).T
#     agg.index.name = 'bin'
#     agg['score'] = (agg['a'] - agg['b']).fillna(0)

#     agg['padj'] = multipletests(agg['pvalue'], method='fdr_bh')[1]
#     agg['slp'] = -np.log10(agg['pvalue']) * np.sign(agg['score']) * (agg['padj'] < 0.05)
#     agg = agg.sort_index()

#     agg['cnt_a'] = agg['cnt_a'].fillna(0).astype(int)
#     agg['cnt_b'] = agg['cnt_b'].fillna(0).astype(int)
#     agg['count'] = np.minimum(agg['cnt_a'], agg['cnt_b'])

#     agg = agg.sort_index()
#     return agg


# def agg_diff_mwu(D, norm=True):
#     "Take LFC of normalized means per bin."
#     from scipy.stats import mannwhitneyu
#     from statsmodels.stats.multitest import multipletests

#     if norm:
#         D.loc[D.a, 'v'] = D.loc[D.a, 'v'] / D.loc[D.a, 'v'].mean()
#         D.loc[D.b, 'v'] = D.loc[D.b, 'v'] / D.loc[D.b, 'v'].mean()

#     gb = D.groupby('bin')
#     rv = {}
#     for name, group in gb:
#         aa = group[group['a']]['v']
#         bb = group[group['b']]['v']
#         row = dict(cnt_a = len(aa),
#                    cnt_b = len(bb))

#         if len(aa) < 3 or len(bb) < 3:
#             row['pval'] = 1
#             row['mwu'] = 0
#             row['lfc'] = 0
#         else:
#             mwu = mannwhitneyu(aa, bb)
#             row['pval'] = mwu.pvalue
#             row['mwu'] = mwu.statistic
#             row['lfc'] = np.log2(aa.mean() / bb.mean())

#         rv[name] = row

#     agg = pd.DataFrame(rv).T
#     agg['pval'] = agg['pval'].clip(1e-200, 1)
#     agg['cnt_a'] = agg['cnt_a'].fillna(0).astype(int)
#     agg['cnt_b'] = agg['cnt_b'].fillna(0).astype(int)
#     agg['padj'] = multipletests(agg['pval'], method='fdr_bh')[1]
#     agg['slp'] = -np.log10(agg['pval']) * np.sign(agg['lfc'])
#     agg.loc[agg['padj'] > 0.05, 'slp'] = 0
#     agg['score'] = agg['slp']
#     return agg


# def hexbinplot(col,
#                adata = None,
#                obsm = None,
#                obs = None,
#                gridsize=16,
#                ax=None,
#                nrm=0.05,
#                brd=0.005,
#                tfs=7,
#                cmap='YlGnBu',
#                vmin=None, vmax=None, vperc=0.05,
#                vzerosym=False,
#                vzerosqueeze=1,
#                edgenrm=0.1,
#                legend=True,
#                legend_fontsize=7,
#                legend_elements: int = 3,
#                lw=0.5,
#                linewidths=None,
#                mask_count=0, mask_alpha=0.5,
#                title=None,
#                title_pad=6,
#                use_rep='X_umap',
#                agg_func=None,

#                marker=None,
#                marker_color='white',
#                marker_lw=4,
#                marker_outline=True,

#                # subset the values to be aggregated.
#                subset=None,
#                mincnt=5,  #less than this number of obs -> transparent

#                binscores=None,

#                diff_a=None,
#                diff_b=None,
#                diff_method='delta',

#                **kwargs):

#     aggdata = None
#     # fix some strange parameter names
#     if linewidths is not None:
#         warnings.warn("Please use `lw=` instead of `linewidths`")
#         lw = linewidths

#     if adata is None:
#         assert obsm is not None
#         assert obs is not None
#     else:
#         if obsm is not None:
#             lg.warning("Ignoring obsm")
#         if obs is not None:
#             lg.warning("Ignoring obs")

#         obsm = adata.obsm
#         obs = adata.obs

#     x = obsm[use_rep][:,0]
#     y = obsm[use_rep][:,1]

#     # To be implemented
#     # assert diff_groupby is None


#     if ax is None:
#         ax = plt.gca()

#     hbargs = a = dict(x=x, y=y, gridsize=gridsize, cmap=cmap,
#                       linewidths=lw, mincnt=0,
#                       edgecolors='black')

#     # check for a categorical column
#     modus = 'score'

#     diff = False

#     if diff_a is not None:
#         if subset is not None:
#             pass # print("Warning - doing a diff and subsetting?")
#         diff = True

#     if col == 'count':
#         if subset is not None:
#             print("not implemented subsetting bin counts")
#             return
#         modus = 'count'

#     elif str(obs[col].dtype) == 'category':
#         assert diff is False  ## not allowed
#         assert subset is None ## also not
#         modus = 'cat'


#     if marker is not None:
#         fig = ax.figure
#         ax2 = fig.add_subplot(111)
#         hb_marker = get_hexbin_categorical(ax2, obs[marker], **hbargs)
#         fig.delaxes(ax2)

#     if modus == 'cat':
#         # calculate ORs for alpha's later onto
#         fig = ax.figure
#         ax2 = fig.add_subplot(111)
#         hb_cator = get_hexbin_categorical(ax2, obs[col], generate_OR=True, **hbargs)
#         fig.delaxes(ax2)


#     # determine how to aggregate
#     if agg_func is None:
#         if diff:
#             if modus == 'count':
#                 agg_func = partial(agg_diff_delta, aggfunc=np.sum)
#             else:
#                 agg_func = agg_diff_delta
#         elif modus == 'count':
#             agg_func = partial(agg_generic, aggfunc=np.sum)
#         else:
#             agg_func = partial(agg_generic, aggfunc="mean")

#     if diff:
#         aggargs = dict(diff_a=diff_a,
#                        diff_b=diff_b,
#                        subset=subset,
#                        agg_func=agg_func)
#     else:
#         aggargs = dict(agg_func=agg_func, subset=subset)


#     if modus == 'cat':
#         hb = get_hexbin_categorical(ax, obs[col], **hbargs)

#     elif modus == 'score':
#         hb = ax.hexbin(C=obs[col], **hbargs)
#         if diff:
#             _raw, aggdata = get_array_diffscore(
#                 obs[col], **hbargs, **aggargs)
#         else:
#             _raw, aggdata = get_array_score(
#                 obs[col], **hbargs, **aggargs)

#         if vmin is None and vmax is None:
#             # calculate un-selected
#             # this is to make plots comparable between
#             # different 'selects'
#             if not diff and subset is not None:
#                 _, aggdata2 = get_array_score(
#                     obs[col], agg_func=agg_func, **hbargs)
#                 vmin = np.quantile(
#                     aggdata2['score'], nrm)
#                 vmax =  np.quantile(
#                     aggdata2['score'], 1-nrm)
#                 # print("set vmin vmax 1a", vmin, vmax)

#                 #aggdata2['score'].max()

#             else:
#                 # there was no subset
#                 vmin = np.quantile(
#                     aggdata['score'], nrm)
#                 vmax =  np.quantile(
#                     aggdata['score'], 1-nrm)
#                 #print("set vmin vmax 1b", vmin, vmax)


#         if diff:
#             if diff_method == 'slp':
#                 # print(aggdata.sort_values(by='slp').tail())
#                 alpha=((aggdata['padj'] <= 0.05).astype(int) * 3 + 1) / 4
#                 hb.set(array=aggdata['slp'], alpha=alpha)
#             elif diff_method == 'delta':
#                 # print(aggdata.sort_values(by='slp').tail())
#                 alpha=((aggdata['padj'] <= 0.05).astype(int) * 3 + 1) / 4
#                 hb.set(array=aggdata['score'], alpha=alpha)
#             elif diff_method == 'delta_all':
#                 hb.set(array=aggdata['score'])
#         else:
#             alpha=((aggdata['count'] >= mincnt).astype(int) * 3 + 1) / 4
#             # alpha=((aggdata['count'] >= mincnt).astype(int) * 3 + 1) / 4
#             hb.set(array=aggdata['score'], alpha=alpha)

#     elif modus == 'count':
#         if not diff:
#             hbargs['mincnt'] = 1
#             hb = ax.hexbin(**hbargs)
#             print("Mean no per bin", hb.get_array().mean())
#         else:
#             o = obs.copy()
#             o['_one'] = 1
#             _raw, aggdata = get_array_diffscore(
#                  o['_one'], **hbargs, **aggargs)
#              #hb = get_hexbin_diffcount(ax, adata.obs,
#              #                         **hbargs, **diffargs)
#     else:
#         raise NotImplementedError()

#     varray = pd.Series(hb.get_array())

#     # (re-) calculate vmin & vmax if required
#     if modus != 'cat':

#         if vmin is None or vmax is None:
#             if vmin is None:
#                 vmin = np.quantile(varray, nrm)
#             if vmax is None:
#                 vmax = np.quantile(varray, 1-nrm)
#             # print("set vmin vmax 2", vmin, vmax)

#         if vzerosym and vmin <= 0 and vmax >= 0:
#             vext = max(abs(vmin), vmax)
#             vmin, vmax = -vext, vext
#             if vzerosqueeze:
#                 vmin *= vzerosqueeze
#                 vmax *= vzerosqueeze


#     hb.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))

#     # ensure proper colormap instance
#     cmap_ = cmap
#     if isinstance(cmap, str):
#         cmap_ = mpl.colormaps[cmap]


#     if marker is not None:

#         valldata = pd.DataFrame(hb_marker.get_offsets())
#         valldata['array'] = pd.Series(hb_marker.get_array())
#         valldata['r0'] = valldata[0].rank(method='dense').astype(int)
#         valldata['r1'] = valldata[1].rank(method='dense').astype(int)


#         #marker_cmap_ = marker_cmap
#         #if isinstance(marker_cmap, str):
#         #    maker_cmap_ = mpl.colormaps[marker_cmap]
#         #marker_face_colors = [marker_cmap_(x) for x in valldata['array'].astype(int)]

#         vertices = hb_marker.get_paths()[0].vertices
#         dirsel = [('right', slice(0,2)),
#                   ('topright', slice(1,3)),
#                   ('topleft', slice(2,4)),
#                   ('left', slice(3,5)),
#                   ('bottomleft', slice(4,6)),
#                   ('bottomright', slice(5,7)),
#                  ]

#         rcutoff = 1 if marker_outline else 2
#         for direction, vsel in dirsel:
#             borders = valldata.apply(find_borders, q=valldata, direction=direction, axis=1)
#             for i, (_, r) in enumerate(borders.items()):
#                 if r >= rcutoff:
#                     xx = vertices[vsel, 0] + valldata.iloc[i][0]
#                     yy = vertices[vsel, 1] + valldata.iloc[i][1]
#                     ax.plot(xx, yy, c=marker_color, zorder=10, lw=marker_lw,)


#     if binscores is not None:
#         hb.set(array=list(binscores))
#         varray = hb.get_array()
#         varray = (varray >= 5).astype(float)
#         varray *= 3
#         varray += 1
#         varray /= 4
#         hb.set(alpha=varray)

#     elif modus == 'cat':
#         # ensure we properly call face colors for category modus
#         # eg - prevent no normalization, ensure values are integers
#         # otherwise it becomes difficult to call
#         def get_cmap(x):
#             if x == -1:
#                 return 'blue'
#             else:
#                 return cmap_(x)

#         face_colors = [cmap_(x) for x in varray.astype(int)]

#         alphas = (pd.Series(hb_cator.get_array() ) > 1.5).astype(float)
#         alphas *= 0.5
#         alphas += 0.5

#         hb.set(array=None, facecolors=face_colors, alpha=alphas)


#     # onto the visuals...
#     #
#     # add a little space around the plot:
#     xmin, xmax = ax.set_xlim()
#     ymin, ymax = ax.set_ylim()
#     xd = brd * (xmax-xmin)
#     yd = brd * (ymax-ymin)
#     ax.set_xlim(xmin-xd, xmax+xd)
#     ax.set_ylim(ymin-yd, ymax+yd)

#     if title is None:
#         title = col
#     ax.set_title(title, pad=title_pad, fontsize=tfs)

#     if legend and modus != 'cat':
#         cnorm=hb.norm
#         no_elements = legend_elements
#         legendpoints = \
#             list(reversed(
#                 [ vmin + ((vmax - vmin) / (no_elements-1) * i)
#                   for i in range(no_elements)]))
#         lem2 = [
#             Patch(facecolor=cmap_(cnorm(i)), edgecolor='k', linewidth=0.3,
#                   label=f"{i:.1g}")
#             for i in legendpoints]

#         legend = ax.legend(handles=lem2, loc='lower left',
#                            handlelength=1.5, labelspacing = 1,
#                            fontsize=legend_fontsize, frameon=False)

#         for handle in legend.legendHandles:
#             handle.set_height(legend_fontsize+3)  # Adjust the height to make it square
#             handle.set_width(legend_fontsize+3)   # Adjust the width to make it square

#     # turn ticks off, remove spines
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)

#     # add per point bin ids & aggregation data per hexin
#     hexbin_ids =  pd.Series(binbin(**hbargs), index=obs.index)
#     hb.hexbin_ids = hexbin_ids
#     hb.aggdata = aggdata


#     return hb


# def hide_axes(ax):
#     """Helper function to hide axis in a gridplot"""
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
