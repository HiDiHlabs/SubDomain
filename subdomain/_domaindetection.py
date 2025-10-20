from collections.abc import Iterable
from functools import partial
from typing import Any

import colorcet as cc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cupy as cp
import seaborn as sns
from jax.lax import conv
from jax.scipy.signal import convolve
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from seaborn.matrix import ClusterGrid
from sklearn.cluster import KMeans
import igraph as ig
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import warnings

_VALID_METHODS = {"circle", "square", "gaussian"} 

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
def _neighborhood(
    arr: jax.Array,
    r: int,
) -> jax.Array:
    """Generates square neighborhood based on the specified method.
    The sum of values within the neighborhood of each pixel is computed via convolution with the appropriate kernel."""
    d = 2 * r + 1
    kernel = jnp.ones((d, d), dtype=arr.dtype)
    return convolve(arr, kernel, mode="same")

@partial(jax.jit, static_argnames="r")
def _circle_neighborhood(
    arr: jax.Array,
    r: int,
) -> jax.Array:
    """Generates circular neighborhood based on the specified method.
    The sum of values within the neighborhood of each pixel is computed via convolution with the appropriate kernel."""
    y, x = jnp.ogrid[-r:r+1, -r:r+1]
    mask = (x**2 + y**2) <= r**2
    kernel = mask.astype(arr.dtype)
    return convolve(arr, kernel, mode="same")

def _gaussian_kernel_1d(r:int, sigma:float):
    """Build a 1D Gaussian kernel with radius r and standard deviation sigma."""
    x = jnp.arange(-r, r + 1)  
    phi_x = jnp.exp(-0.5 * (x / sigma) ** 2) 
    phi_x = phi_x / phi_x.sum() 
    return phi_x

@partial(jax.jit, static_argnames= ("r","sigma"))
def _gaussian_neighborhood(arr: jax.Array, r: int, sigma:float) -> jax.Array:
    """Build the neighborhood consolidation matrix for each pixel using a Gaussian-weighted 
    sum within a neighborhood of radius r, approximated via two 1D convolutions."""
    kernel = _gaussian_kernel_1d(r,sigma)
    # First apply 1D Gaussian convolution along the horizontal direction
    result_row = convolve(arr, kernel[None, :], mode="same")
    # Then apply 1D Gaussian convolution along the vertical direction
    result_final = convolve(result_row, kernel[:, None], mode="same")
    return result_final


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
        self, 
        binsize: int, 
        radius: int, *,
        neighborhood_type: str = "circle", 
        normalize: bool = True,  
        sigma: float = 1.0
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
        neighborhood_type : str, optional
            Method for defining the neighborhood shape. Options:
                - "circle": A circular neighborhood based on Euclidean distance.
                - "square": A square neighborhood with all elements in the kernel.
                - "gaussian": A Gaussian-weighted neighborhood.
        normalize : bool, optional
            Whether to normalize the neighborhood of each bin (L1-norm).
        sigma : float, optional
            Standard deviation for the Gaussian kernel, if `use_gaussian` is True.
        """
        self.binsize = binsize

        if neighborhood_type not in _VALID_METHODS:
            raise ValueError(
                f"Unknown neighborhood_type: {neighborhood_type}. "
                f"Supported types are: {sorted(_VALID_METHODS)}"
            )

        # TODO improve by allocating first?
        if neighborhood_type == "gaussian":
            mtx = jnp.dstack(
            [
                _gaussian_neighborhood(_bin_array(self.label_map == i, binsize), radius,sigma)
                for i in range(self.n_labels)
            ]
            )
        elif neighborhood_type == "square":
            mtx = jnp.dstack(
            [
                _neighborhood(_bin_array(self.label_map == i, binsize), radius)
                for i in range(self.n_labels)
            ]
            )
        elif neighborhood_type == "circle":
            mtx = jnp.dstack(
            [
                _circle_neighborhood(_bin_array(self.label_map == i, binsize), radius)
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
            or [cuml.KMeans][].
        """
        if gpu:
            import cuml

            kmeans = cuml.KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                output_type="numpy",
                n_init=10,
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

    def _build_igraph_with_knn(
        self,
        mtx_valid: np.ndarray,
        n_neighbors: int = 5,
        gpu: bool = False,
        *,
        directed: bool = False,
        metric: str = "cosine",
        radius_knn: float = 1.0,
        use_weights: bool = True,
    ):
        """Build an igraph or cuGraph from neighborhoods using k-NN graph.

        Parameters
        ----------
        mtx_valid : np.ndarray
            Flattened and NaN-filtered neighborhood data.
        n_neighbors : int
            Number of nearest neighbors for each node.
        gpu : bool, optional
            Whether to use the GPU for KNN.
        directed : bool
            Whether to create a directed graph.
        metric : str
            Distance metric for k-NN (e.g., 'euclidean', 'cosine').
        use_weights : bool
            Whether to use edge weights.

        Returns
        -------
        ig.Graph or cugraph.Graph
            Graph object ready for Leiden clustering.
        """

        mode = 'distance' if use_weights else 'connectivity'
        if gpu:
            import cuml
            # GPU mode: fit and compute k-NN graph
            nn_model = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
            nn_model.fit(mtx_valid)
            adjacency = nn_model.kneighbors_graph(mtx_valid, mode=mode)
            adj_coo = adjacency.tocoo()
            row = adj_coo.row
            col = adj_coo.col
            data = adj_coo.data 
            mask = row < col 
            sources = cp.concatenate([row[mask], col[mask]])
            targets = cp.concatenate([col[mask], row[mask]])
            weights = cp.concatenate([data[mask], data[mask]])
        else:
            # CPU mode: fit and compute k-NN graph
            nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, radius=radius_knn)
            nn_model.fit(mtx_valid)
            # adjacency = nn_model.kneighbors_graph(mtx_valid,mode=mode)
            # adjacency = adjacency + adjacency.T
            # adjacency.data = np.minimum(adjacency.data, 1)
            # adjacency.eliminate_zeros()
            # sources, targets, weights = find(adjacency)

            distances, indices = nn_model.kneighbors(mtx_valid)

            n_samples = indices.shape[0]
            n_neighbors = indices.shape[1]

            sources = np.repeat(np.arange(n_samples), n_neighbors - 1)  
            targets = indices[:, 1:].ravel()  
            weights = 1/(1e-10+distances[:, 1:].ravel())
            sources = np.concatenate([sources, targets])
            targets = np.concatenate([targets, sources])
            weights = np.concatenate([weights, weights]) 


        if gpu:
            try:
                import cugraph
                import cudf

                # Build cuGraph from edges
                edge_data = {
                    "source": sources, 
                    "target": targets}
                if use_weights:
                    edge_data["weight"] = weights
                df = cudf.DataFrame(edge_data)

                g = cugraph.Graph(directed=directed)
                g.from_cudf_edgelist(
                    df, 
                    source='source', 
                    destination='target',
                    edge_attr='weight' if use_weights else None)
                return g
            except ImportError:
                warnings.warn("cuGraph unavailable, falling back to CPU.")
                gpu = False  # fallback

        g = ig.Graph(
            n=n_samples,
            edges=list(zip(sources, targets)),
            directed=directed,
            edge_attrs={"weight": weights} if use_weights else {}
        )
        return g


    def leiden_cluster_neighborhoods(
            self,
            resolution: float = 1.0,
            *,
            n_neighbors: int = 15,
            directed: bool = False,
            metric: str = "cosine",
            radius_knn: float = 1.0,
            gpu: bool = False,
            use_weights: bool = True,
            **kwargs
        ):
        """Cluster neighborhoods using Leiden algorithm with k-NN graph.

        Parameters
        ----------
        resolution : float
            Leiden resolution parameter (higher = more clusters).
        n_neighbors : int
            Number of neighbors for k-NN graph.
        directed : bool
            Whether to use directed k-NN graph.
        metric : str
            Distance metric for k-NN.
        use_weights : bool
            Whether to use edge weights.
        gpu : bool
            Whether to use GPU-accelerated Leiden (requires cuGraph).
        kwargs
            Additional Leiden parameters.
        """
        # Flatten and filter out NaN values
        mtx_flat = _flatten_2d(self.neighborhoods)
        valid_mask = ~jnp.isnan(mtx_flat).any(axis=1)
        mtx_valid = mtx_flat[valid_mask]

        # Build k-NN graph
        graph = self._build_igraph_with_knn(
            mtx_valid=mtx_valid,
            n_neighbors=n_neighbors,
            radius_knn=radius_knn,
            gpu=gpu,
            directed=directed,
            metric=metric,
            use_weights=use_weights,
        )

        # Run Leiden clustering
        if gpu:
            try:
                import cugraph
            except ImportError:
                warnings.warn("cuGraph not available, falling back to CPU")
            if isinstance(graph, cugraph.Graph):
                parts,_ = cugraph.leiden(graph, resolution=resolution,max_iter=100)
                categories =parts.sort_values("vertex")["partition"].values.get()
        else:
            categories = graph.community_leiden(
                weights=graph.es["weight"] if use_weights else None,
                resolution=resolution,
                max_iter=100,
                **kwargs
            ).membership

        # Map results back to original grid
        domain = np.full(mtx_flat.shape[0], -1, dtype=np.int16)
        domain[valid_mask] = categories 
        self.domains = domain.reshape(self.neighborhoods.shape[:2])
        self.n_domains = len(np.unique(categories))

    def gaussian_mixture_neighborhoods(
            self, 
            num_components: int = 1, 
            *, 
            gpu: bool = False, 
            random_state: int = 1, 
            **kwargs
    ):
        """Cluster the aggregated neighborhoods using a Gaussian Mixture Model (GMM).

        Parameters
        ----------
        n_components : int, optional
            Number of mixture components (clusters), by default 1.
        gpu : bool, optional
            If True, use GPU-accelerated GMM based on PyTorch; otherwise use scikit-learn, by default False.
        random_state : int, optional
            Random seed for reproducibility, by default 1.
        kwargs : dict
            Additional keyword arguments passed to the GMM model.

        Updates
        -------
        self.domains : np.ndarray
            Cluster labels reshaped to the original neighborhood grid.
        self.n_domains : int
            Number of clusters (same as n_components).
        """
        if gpu:
            from torchgmm.bayes import GaussianMixture as TorchGaussianMixture
            import torch

            # Initialize GPU GMM
            gm = TorchGaussianMixture(
                num_components=num_components,
                **kwargs
            )
        else:
            gm = GaussianMixture(
                n_components=num_components,
                random_state=random_state,
                **kwargs
            )

        # Flatten the neighborhood matrix
        mtx_flat = _flatten_2d(self.neighborhoods)
        not_nan = ~jnp.isnan(mtx_flat).any(axis=1)
        domain = np.full(mtx_flat.shape[0], -1, dtype=np.int16)

        if gpu:
            # Convert to torch tensor and move to GPU
            tensor_data = torch.tensor(np.copy(mtx_flat[not_nan]), dtype=torch.float32).cuda()
            preds = gm.fit_predict(tensor_data)
            preds = preds.cpu().numpy()
        else:
            preds = gm.fit_predict(mtx_flat[not_nan])

        domain[not_nan] = preds
        self.domains = domain.reshape(self.neighborhoods.shape[:2])
        self.n_domains = num_components





    def identify_domains(
        self,
        binsize: int = 8,
        radius: int = 10,
        neighborhood_type: str = "circle",
        n_clusters: int = 10,
        sigma: float = 1.0,
        *,
        gpu: bool = False,
        random_state: int = 1,
        clustering_method: str = "kmeans",
        resolution: float = 1.0,
        n_neighbors: int = 15,
        radius_knn: float = 1.0,
        directed: bool = False,
        metric: str = "cosine",
        use_weights: bool = True,
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
        neighborhood_type : str, optional
            Method for defining the neighborhood shape. Options:
            - "circle": A circular neighborhood based on Euclidean distance.
            - "square": A square neighborhood with all elements in the kernel.
            - "gaussian": A Gaussian-weighted neighborhood.
        n_clusters : int
            Number of clusters for k-means.
        sigma : float, optional
        gpu: bool, optional
            Whether to use the GPU for KMeans clustering. The neighborhood aggregation will
            run by default on GPU if available.
        random_state : int, optional
            Random state for reproducibility.
        clustering_method : str, optional
            The clustering method to use. Options are:
            - 'kmeans': Use KMeans clustering.
            - 'leiden': Use Leiden clustering.
            - 'gmm': Use Gaussian Mixture Model clustering.
        resolution : float, optional
            Resolution parameter for Leiden clustering.
        n_neighbors : int, optional
            Number of neighbors for Leiden clustering.
        directed : bool, optional
            Whether the graph is directed for Leiden clustering.
        metric : str, optional
            The distance metric for Leiden clustering.
        use_weights : bool, optional
            Whether to use weights in Leiden clustering.
        normalize : bool, optional
            Whether to normalize the neighborhood of each bin (L1-norm).
        kwargs
            Other keyword arguments will be passed to [sklearn.cluster.KMeans][]
            or [cuml.KMeans][].
        """
        self.calculate_neighborhoods(binsize, radius,neighborhood_type=neighborhood_type,sigma=sigma)
        if clustering_method == "leiden":
            self.leiden_cluster_neighborhoods(
                resolution=resolution,
                n_neighbors=n_neighbors,
                radius_knn=radius_knn,
                directed=directed,
                metric=metric,
                gpu=gpu,
                use_weights=use_weights)
        elif clustering_method == "kmeans":
            self.cluster_neighborhoods(
                n_clusters, gpu=gpu, random_state=random_state, **kwargs
            )
        elif clustering_method == "gmm":
            self.gaussian_mixture_neighborhoods(
                num_components=n_clusters,
                gpu=gpu,
                **kwargs
            )
        else:
            raise ValueError(f"Invalid clustering method '{clustering_method}'. Choose either 'kmeans' or 'leiden' or 'gmm'.")

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
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))

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
        fig.subplots_adjust(wspace=0.3) 
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
