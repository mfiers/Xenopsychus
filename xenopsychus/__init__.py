"""Xenopsychus - hexbins for scanpy


chatgpt thought Xenopsychus was the latin name for the angler fish -
it does not seem to be - but who cares...

"""

from functools import partial
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


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
    tx,ty = x, y

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
    padding = 1.e-9 * (xmax - xmin)
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
    i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1),
                  ix1 * ny1 + iy1 + 1, 0)
    i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2),
                  ix2 * ny2 + iy2 + 1, 0)

    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = (d1 < d2)
    
    # here I take over again
    # calculate per orignal data point the
    # final hexbin ID.
    idb1 = pd.Series(i1) - 1
    idb2 = pd.Series(i2) + (nx1 * ny1 - 1)
    idb1.loc[~bdist] = idb2.loc[~bdist]
    return list(idb1)


def get_array_diffscore(C, diff_a, diff_b, diff_groupby,
                        agg_func, **hbargs):
    """
    Prepare the array for diff-score calculation
    Also, execute the agg_func & 
    """
    if diff_b is None:
        diff_b = ~diff_a

    bins = binbin(**hbargs)
    
    D = pd.DataFrame(dict(
        a=diff_a, b=diff_b, v=C,
        bin=list(bins)))

    arr = agg_func(D).sort_index()
    return D, arr


def get_array_score(C, agg_func,
                    **hbargs):
    """
    Prepare the array for score calculation
    And execute the agg_func
    """
    bins = binbin(**hbargs)
    
    D = pd.DataFrame(dict(
        v=C, bin=bins))
    arr = agg_func(D).sort_index()
    return D, arr


def get_hexbin_diffcount(ax, obs, diff_a, diff_b, diff_groupby,
                         diff_score,
                         **hbargs):

    sel_a = pd.Series(0, index=obs.index)
    sel_b = pd.Series(0, index=obs.index)

    sel_a.loc[obs.query(diff_a).index] = 1

    if diff_b is not None:
        sel_b.loc[obs.query(diff_b).index] = 1
        assert (sel_a & sel_b).sum() == 0
    else:
        sel_b.loc[sel_a[sel_a != 0b01].index] = 1

        
    hb_diff_a = ax.hexbin(
        C=sel_a, reduce_C_function=np.sum,
        visible=False, **hbargs)
    hb = hb_diff_b = ax.hexbin(
        C=sel_b, reduce_C_function=np.sum,
        visible=True, **hbargs)

    arr_a = pd.Series(hb_diff_a.get_array())
    arr_b = pd.Series(hb_diff_b.get_array())
        
    D = pd.DataFrame(dict(a=arr_a, b=arr_b)).fillna(0).astype(int)
    sum_a, sum_b = D.sum()
    allfrac = sum_a / (sum_a + sum_b)

    def bt(row):
        if row.sum() == 0:
            return 1.
        rv = binomtest(row[0], row.sum(), p=allfrac)
        return rv.pvalue
    
    lor = np.log2(  ((D['a'] / D.sum(1)) / allfrac ))
    lor = lor.fillna(0)
    lor = lor.replace([np.inf, -np.inf], 0)

    if diff_score == 'slp':
        pvals = D.apply(bt, axis=1)
        D['lor'] = lor
        D['pval'] = pvals
        D['padj'] = multipletests(pvals, method='fdr_bh')[1]
        
        D['slp'] = -1 * np.log10(D['padj']) * np.sign(D['lor'])
        
        D.loc[D['padj']>0.05, 'slp'] = 0

        hb.set(array=D['slp'])
    else:
        hb.set(array=lor)
    return hb


def get_hexbin_categorical(ax, C, **hbargs):

    def _most_abundant(values):
        # return most abundant value for when we'er in category modus
        
        # find frequency of each value
        values, counts = np.unique(values, return_counts=True)
        # display value with highest frequency
        rv = values[counts.argmax()]
        return int(rv)

    try:
        C = C.astype(int)
    except:
        C = C.cat.codes

    return ax.hexbin(C=C, 
                   reduce_C_function=_most_abundant,
                   **hbargs)   
    

def agg_generic(D, aggfunc=np.mean):
    agg = pd.DataFrame(dict(
        score = D.groupby('bin')['v'].agg(aggfunc),
        count = D.groupby('bin')['v'].count()
        ))
    return agg
    
def agg_diff_lfc(D, norm=False, aggfunc=np.mean):
    "Take LFC of normalized means per bin."

    if norm:
        D.loc[D.a, 'v'] = D.loc[D.a, 'v'] / D.loc[D.a, 'v'].mean()
        D.loc[D.b, 'v'] = D.loc[D.b, 'v'] / D.loc[D.b, 'v'].mean()
        
    agg = pd.DataFrame(dict(
        a = D[D['a']].groupby('bin')['v'].agg(aggfunc),
        b = D[D['b']].groupby('bin')['v'].agg(aggfunc),
        cnt_a = D[D['a']].groupby('bin')['v'].count(),
        cnt_b = D[D['b']].groupby('bin')['v'].count(),
    ))

    agg['cnt_a'] = agg['cnt_a'].fillna(0).astype(int)
    agg['cnt_b'] = agg['cnt_b'].fillna(0).astype(int)
    agg['count'] = np.minimum(agg['cnt_a'], agg['cnt_b'])
    
    agg['score'] = (np.log2(agg['a'] / agg['b']))\
        .replace([-np.inf, np.inf], 0)\
        .fillna(0)

    return agg


def agg_diff_mwu(D, norm=True):
    "Take LFC of normalized means per bin."
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    
    if norm:
        D.loc[D.a, 'v'] = D.loc[D.a, 'v'] / D.loc[D.a, 'v'].mean()
        D.loc[D.b, 'v'] = D.loc[D.b, 'v'] / D.loc[D.b, 'v'].mean()

    gb = D.groupby('bin')
    rv = {}
    for name, group in gb:
        aa = group[group['a']]['v']
        bb = group[group['b']]['v']
        row = dict(cnt_a = len(aa),
                   cnt_b = len(bb))
        
        if len(aa) < 3 or len(bb) < 3:
            row['pval'] = 1
            row['mwu'] = 0
            row['lfc'] = 0
        else:
            mwu = mannwhitneyu(aa, bb)
            row['pval'] = mwu.pvalue
            row['mwu'] = mwu.statistic
            row['lfc'] = np.log2(aa.mean() / bb.mean())
    
        rv[name] = row

    agg = pd.DataFrame(rv).T
    agg['pval'] = agg['pval'].clip(1e-200, 1)
    agg['cnt_a'] = agg['cnt_a'].fillna(0).astype(int)
    agg['cnt_b'] = agg['cnt_b'].fillna(0).astype(int)
    agg['padj'] = multipletests(agg['pval'], method='fdr_bh')[1]
    agg['slp'] = -np.log10(agg['pval']) * np.sign(agg['lfc'])
    agg.loc[agg['padj'] > 0.05, 'slp'] = 0
    agg['score'] = agg['slp']
    #print(agg)
    return agg


def hexbinplot(adata,
               col,
               gridsize=16,
               ax=None,
               nrm=0.05, 
               brd=0.005,
               tfs=7,
               cmap='YlGnBu', 
               vmin=None, vmax=None,
               vzerosym=True,
               edgenrm=0.1,
               legend_fontsize=7, linewidths=0.5, 
               mask_count=0, mask_alpha=0.5, 
               title=None,
               use_rep='X_umap',
               agg_func=None,
               
               diff_a=None,
               diff_b=None,
               diff_groupby=None,

               **kwargs):

    
    x = adata.obsm[use_rep][:,0]
    y = adata.obsm[use_rep][:,1]

    # To be implemented
    assert diff_groupby is None

    
    if ax is None:
        ax = plt.gca()
            
    hbargs = a = dict(x=x, y=y, gridsize=gridsize, cmap=cmap, 
             linewidths=linewidths, mincnt=0,
             edgecolors='black')
            
    # check for a categorical column
    modus = 'score'
    diff = False

    if diff_a is not None:
        diff = True

    if col == 'count':
        modus = 'count'
        
    elif str(adata.obs[col].dtype) == 'category':
        assert diff is False  ## not allowed 
        modus = 'cat'

        
    # determine how to aggregate
    if agg_func is None:
        if diff:
            agg_func = agg_diff_lfc
        elif modus == 'count':
            agg_func = partial(agg_generic, aggfunc=np.sum)
        else:
            agg_func = partial(agg_generic, aggfunc=np.mean)

    if diff:
        aggargs = dict(diff_a=diff_a,
                       diff_b=diff_b,
                       diff_groupby=diff_groupby,
                       agg_func=agg_func)
    else:
        aggargs = dict(agg_func=agg_func)

    if modus == 'cat':
        hb = get_hexbin_categorical(ax, adata.obs[col], **hbargs)
        
    elif modus == 'score':
        hb = ax.hexbin(C=adata.obs[col], **hbargs)
        if diff:
            _raw, aggdata = get_array_diffscore(
                adata.obs[col], **hbargs, **aggargs)            
        else:
            _raw, aggdata = get_array_score(
                adata.obs[col], **hbargs, **aggargs)
        hb.set(array=aggdata['score'])
        
    elif modus == 'count':
        if not diff:
            hb = ax.hexbin(**hbargs)
        else:
            hb = get_hexbin_diffcount(ax, adata.obs,
                                      **hbargs, **diffargs)
    else:
        raise NotImplementedError()
    
    varray = pd.Series(hb.get_array())

    # (re-) calculate vmin & vmax if required
    if modus != 'cat':

        if vmin is None or vmax is None:
            if vmin is None:
                vmin = np.quantile(varray, nrm)
            if vmax is None:
                vmax = np.quantile(varray, 1-nrm)

            #if vzerosym and vmin <= 0 and vmax >= 0:
            #    vext = max(abs(vmin), vmax)
            #    vmin, vmax = -vext, vext
        
    hb.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))

    # ensure proper colormap instance
    cmap_ = cmap
    if isinstance(cmap, str):
        cmap_ = mpl.colormaps[cmap]

    
    if modus == 'cat':
        # ensure we properly call face colors for category modus
        # eg - prevent no normalization, ensure values are integers
        # otherwise it becomes difficult to call
        
        face_colors = [cmap_(x) for x in varray.astype(int)]
        hb.set(array=None, facecolors=face_colors)

    # onto the visuals...
    #
    # add a little space around the plot:
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    xd = brd * (xmax-xmin)
    yd = brd * (ymax-ymin)
    ax.set_xlim(xmin-xd, xmax+xd)
    ax.set_ylim(ymin-yd, ymax+yd)

    if title is None:
        title = col
    ax.set_title(title, fontsize=tfs)

    if modus != 'cat':
        cnorm=hb.norm
        legend_elements = [
            Patch(facecolor=cmap_(cnorm(vmin)), edgecolor='k', 
                  linewidth=0.3, label=f"{vmin:.1f}"),
            Patch(facecolor=cmap_(cnorm(vmax)), edgecolor='k', 
                  linewidth=0.3, label=f"{vmax:.1f}"), ]
    
        legend = ax.legend(handles=legend_elements, loc='lower left', 
                           handlelength=0.9, 
                           fontsize=legend_fontsize, frameon=False)
    
        for handle in legend.legend_handles:
            handle.set_height(7)  # Adjust the height to make it square
            handle.set_width(7)   # Adjust the width to make it square

    # turn ticks off, remove spines
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
  
    return hb
