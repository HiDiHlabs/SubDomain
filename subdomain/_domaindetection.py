from collections.abc import Iterable
from functools import partial
from typing import Any

import colorcet as cc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax.lax import conv
from jax.scipy.signal import convolve
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from seaborn.matrix import ClusterGrid
from sklearn.cluster import KMeans


@partial(jax.jit, static_argnames="s")
def _bin_array(arr: jax.Array, s: int) -> jax.Array:
    """Bins an array using a box kernel of size s using strided convolution."""
    kernel = jnp.ones((1, 1, s, s), arr.dtype)  # Box kernel
    bins = conv(
        arr.reshape(1, 1, *arr.shape), kernel, window_strides=(s, s), padding="SAME"
    )
    return jnp.squeeze(bins)


# gaussian i) via FFT but kernel likely not large enough ii) via 2x 1D convolutions


@partial(jax.jit, static_argnames="r")
def _neighborhood(arr: jax.Array, r: int) -> jax.Array:
    """Build the neighborhood consolidation matrix for each pixel as the sum of the
    neighborhood with radius r."""
    d = 2 * r + 1
    kernel = jnp.ones((d, d), arr.dtype)
    return convolve(arr, kernel, mode="same")


@jax.jit
def _flatten_2d(mtx: jax.Array) -> jax.Array:
    return mtx.reshape(mtx.shape[0] * mtx.shape[1], *mtx.shape[2:])


_HEATMAP_KWARGS: "dict[str, Any]" = dict(
    vmin=0,
    row_cluster=False,
    cbar_pos=(1, 0.1, 0.03, 0.7),
    cbar_kws=dict(label="frequency"),
    dendrogram_ratio=(0.01, 0.1),
    yticklabels=False,
)


class SubDomain:
    """Analyse domains based on labels in a 2D grid.

    Parameters
    ----------
    label_map : numpy.ndarray | jax.Array
        An integer array where all positive values correspond to a specific cell type
        and negative values are background.
    label_name : str, optional
        Name of the labels.
    labels : collections.abc.Iterable[str] | None, optional
        Names corresponding to each label in `label_map`.

    Raises
    ------
    ValueError
        If the length of `labels` does not match the number of labels in `label_map`.

    Attributes
    ----------
    SubDomain.label_map : numpy.ndarray | jax.Array
        2D labeled grid.
    SubDomain.n_labels : int
        Number of different categories in `label_map` (excluding background).
    SubDomain.label_name : str
        Name of the labels.
    SubDomain.labels : str
        Names corresponding to each label in `label_map`.
    SubDomain.neighborhoods : jax.Array
        The consolidated neighborhoods after binning.
    SubDomain.binsize : int
        Size of each domain bin.
    SubDomain.domains : numpy.ndarray
        The assigned domain for each bin.
    SubDomain.n_domains : int
        Number of domains.
    """

    def __init__(
        self,
        label_map: np.ndarray | jax.Array,
        /,
        *,
        label_name: str = "celltype",
        labels: Iterable[str] | None = None,
    ):
        self.label_map = label_map
        self.n_labels: int = int(self.label_map.max()) + 1
        self.label_name = label_name

        # TODO validate the unique indices

        if labels is not None:
            labels = list(labels)
            if len(labels) != self.n_labels:
                raise ValueError(
                    "Length of `labels` must match the number of labels in `label_map`."
                )
        self.labels = labels

    def calculate_neighborhoods(
        self, binsize: int, radius: int, *, normalize: bool = True
    ):
        """Calculate the neighborhoods.

        The label map is binned and subsequently the neighborhood in terms of frequency
        per label calculated for each bin.

        Parameters
        ----------
        binsize : int
            Size to bin the labeled grid by.
        radius : int
            Radius for the neighborhood aggregation. The size of the neighborhood will be
            `2 * binsize * (radius + 1)`
        normalize : bool, optional
            Whether to normalize the neighborhood of each bin (L1-norm).
        """
        self.binsize = binsize

        # TODO improve by allocating first?
        mtx = jnp.dstack(
            [
                _neighborhood(_bin_array(self.label_map == i, binsize), radius)
                for i in range(self.n_labels)
            ]
        )

        if normalize:
            l1_norm = mtx.sum(axis=2)
            # Avoid division by zero
            mtx /= l1_norm.at[l1_norm == 0].set(1e-10)[:, :, None]
            # set to nan
            mtx = mtx.at[l1_norm == 0, :].set(jnp.nan)
        self.neighborhoods = mtx

    def cluster_neighborhoods(
        self, n_clusters: int, *, gpu: bool = False, random_state: int = 1, **kwargs
    ):
        """Cluster the aggregated neighborhoods.

        Assigns a domain (cluster) to each bin in the calculated neighborhoods (requires
        to first run [subdomain.SubDomain.calculate_neighborhoods][]).

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        gpu : bool, optional
            Whether to use the GPU for KMeans clustering.
        random_state : int, optional
            Random state for reproducibility.
        kwargs
            Other keyword arguments will be passed to [sklearn.cluster.KMeans][]
            or [cuml.cluster.KMeans][].
        """
        if gpu:
            import cuml

            kmeans = cuml.KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                output_type="numpy",
                **kwargs,
            )
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)

        mtx_flat = _flatten_2d(self.neighborhoods)
        not_nan = ~jnp.isnan(mtx_flat).any(axis=1)

        domain = np.full(mtx_flat.shape[0], -1, dtype=np.int16)
        domain[not_nan] = kmeans.fit_predict(mtx_flat[not_nan])
        self.domains = domain.reshape(self.neighborhoods.shape[:2])
        self.n_domains = n_clusters

    def identify_domains(
        self,
        binsize: int = 8,
        radius: int = 10,
        n_clusters: int = 10,
        *,
        gpu: bool = False,
        random_state: int = 1,
        **kwargs,
    ):
        """Identify domains from labeled grid.

        This is a wrapper around [subdomain.SubDomain.calculate_neighborhoods][] and
        [subdomain.SubDomain.cluster_neighborhoods][].

        If the neighborhood has already been calculated (and the parameters do not need
        to be changed) it is more efficient to just cluster the domains rather than
        recalculating the neighborhoods.

        Parameters
        ----------
        binsize : int
            Size to bin the labeled grid by.
        radius : int
            Radius for the neighborhood aggregation. The size of the neighborhood will be
            `2 * binsize * (radius + 1)`
        n_clusters : int
            Number of clusters for k-means.
        gpu: bool, optional
            Whether to use the GPU for KMeans clustering. The neighborhood aggregation will
            run by default on GPU if available.
        random_state : int, optional
            Random state for reproducibility.
        kwargs
            Other keyword arguments will be passed to [sklearn.cluster.KMeans][]
            or [cuml.cluster.KMeans][].
        """
        self.calculate_neighborhoods(binsize, radius)

        self.cluster_neighborhoods(
            n_clusters, gpu=gpu, random_state=random_state, **kwargs
        )

    def domain_neighborhoods(self) -> pd.DataFrame:
        """Average neighborhood of the domains.

        Returns
        -------
        pandas.DataFrame
            Average neighborhood.
        """
        neighbor_fractions = (
            pd.DataFrame(_flatten_2d(self.neighborhoods), columns=self.labels)
            .assign(domain=self.domains.ravel())
            .loc[lambda df: df["domain"].ge(0)]
            .groupby("domain")
            .agg("mean")
        )
        neighbor_fractions.columns.name = self.label_name
        return neighbor_fractions

    def domain_composition(self) -> pd.DataFrame:
        """Label composition of each domain.

        Returns
        -------
        pandas.DataFrame
            Label composition.
        """
        name = self.label_name

        domain_composition = (
            pd.DataFrame(
                {
                    name: self.label_map.ravel(),
                    "domain": self.rescale_domain_map().ravel(),
                }
            )
            .loc[lambda df: df[name].ge(0)]
            .groupby(["domain", name])
            .size()
        )
        domain_composition /= domain_composition.groupby("domain").transform("sum")
        return (
            domain_composition.to_frame("fraction")
            .reset_index()
            .pivot(index="domain", columns=name, values="fraction")
            .fillna(0)
        )

    def rescale_domain_map(self) -> np.ndarray:
        """Rescale domain map to original labeled grid size i.e. prior to binning."""
        rescaled_domains = np.repeat(
            np.repeat(self.domains, self.binsize, axis=0), self.binsize, axis=1
        )[: self.label_map.shape[0], : self.label_map.shape[1]]
        return rescaled_domains

    def plot_domains(
        self,
        domain_palette=cc.glasbey_dark,
        label_palette=cc.glasbey_light,
        *,
        scale: tuple[float, str] | None = None,
        **kwargs,
    ) -> Figure:
        """Spatial plot of domains and labeled grid.

        Parameters
        ----------
        domain_palette
            Palette to use for the domain plot. Must be a valid argument for
            [seaborn.color_palette][]]
        label_palette
            Palette to use for the labeled grid plot. Must be a valid argument for
            [seaborn.color_palette][]]
        scale : tuple[float, str] | None
            Size of a pixel in the original labeled grid as a tuple of the value and
            the unit (must be one of nm, um, ...) e.g. `(5, 'um')`.
        kwargs
            Other keyword arguments are passed to `matplotlib-scalebar.ScaleBar`

        Returns
        -------
        matplotlib.figure.Figure
        """

        def _color_lut(
            img: np.ndarray | jax.Array, cmap: list[tuple[float, ...]]
        ) -> jax.Array:
            return jnp.take(jnp.array(cmap), img + 1, axis=0)

        def _plot_image(
            ax: Axes, im, palette, n: int, title: str, labels: Iterable | None = None
        ):
            if labels is None:
                labels = range(n)
            cmap = sns.color_palette(palette, n)
            legend = [Patch(color=c, label=lbl) for c, lbl in zip(cmap, labels)]

            ax.imshow(_color_lut(im, [(0, 0, 0)] + cmap), origin="lower")
            ax.legend(
                handles=legend,
                ncols=-(n // -10),
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
            ax.set(title=title)

        fig, axs = plt.subplots(nrows=2)

        _plot_image(
            axs[0],
            self.label_map.T,
            label_palette,
            self.n_labels,
            "Labels",
            self.labels,
        )
        _plot_image(axs[1], self.domains.T, domain_palette, self.n_domains, "Domains")

        if scale is not None:
            axs[0].add_artist(ScaleBar(*scale, **kwargs))
        fig.tight_layout()
        return fig

    def plot_neighborhood_heatmap(
        self, *, palette=cc.glasbey_dark, **kwargs
    ) -> ClusterGrid:
        """Heatmap of the label enrichment of the domains.

        Parameters
        ----------
        palette : str, optional
            A valid argument for [seaborn.color_palette][]
        kwargs
            Other keyword arguments are passed to [seaborn.clustermap][]

        Returns
        -------
        seaborn.ClusterGrid
            Heatmap returned from [seaborn.clustermap][]
        """

        domains_flat = self.domains.ravel()
        # remove background
        not_background = domains_flat >= 0
        domains_flat = domains_flat[not_background]  # type: ignore

        order = np.argsort(domains_flat)

        domain_ids = np.unique(domains_flat)
        lut = dict(zip(domain_ids, sns.color_palette(palette, len(domain_ids))))

        g = sns.clustermap(
            _flatten_2d(self.neighborhoods)[not_background][order],
            row_colors=pd.Series(domains_flat[order]).map(lut).to_numpy(),
            **(_HEATMAP_KWARGS | kwargs),
        )
        g.ax_row_dendrogram.set_visible(False)
        g.ax_heatmap.set(xlabel=self.label_name)

        assert g.ax_row_colors is not None
        g.ax_row_colors.set_ylabel("bin")
        g.ax_row_colors.set_xlabel("domain", rotation="vertical")

        # Add black border to the colorbar
        assert g.ax_cbar is not None
        for spine in g.ax_cbar.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)
        return g
