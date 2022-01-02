"""
Method and parts of the code come from https://github.com/hfawaz/cd-diagram.
"""
# standard imports
import math

# external imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx

# internal imports
from .utils import from_result_dict_to_long_format
from .rank_analysis import (
    wilcoxon_holm,
    mean_rank,
)

matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Arial"


def graph_ranks(
    mean_ranks,
    wilcoxon_test_results,
    lowv=None,
    highv=None,
    width=6,
    textspace=1,
    reverse=False,
    labels=False,
    title=None,
    verbose=False,
    **kwargs
):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.
    Needs matplotlib to work.
    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.
    Args:
        mean_ranks (pd.Series): average ranks of methods. Index should be names of methods.
        wilcoxon_test_results (list of WilcoxonTestResult obejcts): results of the paired wilcoxon tests.
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        labels (bool, optional): if set to `True`, the calculated avg rank
            values will be displayed
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    # Internal functions definitions (first set)
    def nth(liste, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(liste, n)
        return [a[n] for a in liste]

    def lloc(liste, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(liste[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.
        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]
        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    # variable initialization
    sums = mean_ranks.values
    nnames = mean_ranks.index.array
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(ssums), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    def hfl(liste):
        return [a * hf for a in liste]

    def wfl(liste):
        return [a * wf for a in liste]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(liste, color="k", **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(liste, 0)), hfl(nth(liste, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=2)

    for a in range(lowv, highv + 1):
        text(
            rankpos(a),
            cline - tick / 2 - 0.05,
            str(a),
            ha="center",
            va="bottom",
            size=16,
        )

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace - 0.1, chei),
            ],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + 0.3,
                chei - 0.075,
                format(ssums[i], ".4f"),
                ha="right",
                va="center",
                size=10,
            )
        text(
            textspace - 0.2,
            chei,
            filter_names(nnames[i]),
            ha="right",
            va="center",
            size=16,
        )

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(ssums[i], ".4f"),
                ha="left",
                va="center",
                size=10,
            )
        text(
            textspace + scalewidth + 0.2,
            chei,
            filter_names(nnames[i]),
            ha="left",
            va="center",
            size=16,
        )

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for liste, r in lines:
            line(
                [
                    (rankpos(ssums[liste]) - side, start),
                    (rankpos(ssums[r]) + side, start),
                ],
                linewidth=linewidth_sign,
            )
            start += height
            if verbose:
                print("drawing: ", liste, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(wilcoxon_test_results, nnames)
    i = 1
    achieved_half = False
    if verbose:
        print("nnames:", nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        if verbose:
            print("clq:", clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and not (achieved_half):
            start = cline + 0.25
            achieved_half = True
        line(
            [
                (rankpos(ssums[min_idx]) - side, start),
                (rankpos(ssums[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
        )
        start += height

    font = {
        "family": "sans-serif",
        "color": "black",
        "weight": "normal",
        "size": 22,
    }
    if title:
        ax.set_title(title, fontdict=font, y=0.9, x=0.5)

    return ax


def form_cliques(wilcoxon_test_results, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for result in wilcoxon_test_results:
        if result.is_significant is False:
            i = np.where(nnames == result.name_1)[0][0]
            j = np.where(nnames == result.name_2)[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagrams_from_result_dict(result_dict, alpha, labels=False):
    result_long_df = from_result_dict_to_long_format(result_dict)
    metrics = [m for m in result_long_df["metric"].unique() if "time" not in m]

    for metric in metrics:
        # filter evaluator according to number
        wilcoxon_holm_p_values = wilcoxon_holm(result_long_df, metric, alpha)
        mean_ranks = mean_rank(result_long_df, metric)

        graph_ranks(
            mean_ranks,
            wilcoxon_holm_p_values,
            reverse=True,
            width=9,
            textspace=1.5,
            labels=labels,
            title=metric,
        )
        plt.savefig(metric + " cd-diagram.png", bbox_inches="tight")
