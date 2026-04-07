import numpy as np

from .base import BaseScorer


class MahalanobisScorer(BaseScorer):
    """
    Anomaly score = Mahalanobis distance from the normal class distribution.

    A random projection to n_components dimensions is applied before fitting
    to reduce the covariance matrix to a tractable size and improve numerical
    stability.

    fit() accepts both:
      - (N, D)            → image-level / CLS token mode
      - (N, n_patches, D) → patch-level mode (per-position statistics)

    In patch-level mode, per-position means and inverse covariances are
    precomputed and cached in fit() so that score_patches() is pure inference.
    """

    name = "mahalanobis"

    def __init__(self, reg: float = 1e-2, n_components: int = 128):
        self.reg = reg
        self.n_components = n_components
        self._patch_mode = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_proj(self, D: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((D, self.n_components)) / np.sqrt(self.n_components)

    def _fit_cls(self, train_embeddings: np.ndarray) -> None:
        """Image-level fit. Input: (N, D)"""
        self._proj = self._make_proj(train_embeddings.shape[1])
        projected = train_embeddings @ self._proj          # (N, n_components)
        self._mean = projected.mean(axis=0)                # (n_components,)
        cov = np.cov(projected, rowvar=False)
        cov += np.eye(cov.shape[0]) * self.reg
        self._inv_cov = np.linalg.inv(cov)                 # (n_components, n_components)

    def _fit_patches(self, train_patches: np.ndarray) -> None:
        """Patch-level fit. Input: (N_train, n_patches, D)"""
        N_train, n_patches, D = train_patches.shape
        self._proj = self._make_proj(D)

        train_proj = train_patches @ self._proj            # (N_train, n_patches, n_components)

        # Per-position statistics — cached for score_patches()
        self._means = train_proj.mean(axis=0)              # (n_patches, n_components)
        centered = train_proj - self._means                # (N_train, n_patches, n_components)
        covs = np.einsum("npc,npd->pcd", centered, centered) / (N_train - 1)
        covs += np.eye(self.n_components) * self.reg
        self._inv_covs = np.linalg.inv(covs)               # (n_patches, n_components, n_components)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_embeddings: np.ndarray) -> None:
        """
        Accepts (N, D) for CLS mode or (N, n_patches, D) for patch mode.
        Patch mode precomputes all per-position statistics here so that
        score_patches() is pure inference with no redundant computation.
        """
        if train_embeddings.ndim == 3:
            self._patch_mode = True
            self._fit_patches(train_embeddings)
        else:
            self._patch_mode = False
            self._fit_cls(train_embeddings)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        CLS token scoring. Requires fit() with (N, D) input.
        Returns (N,) anomaly scores.
        """
        if self._patch_mode:
            raise RuntimeError(
                "Scorer was fitted in patch mode. "
                "Call fit() with (N, D) CLS embeddings for image-level scoring."
            )
        if not hasattr(self, '_inv_cov'):
            raise RuntimeError("Call fit() before score().")

        projected = test_embeddings @ self._proj           # (N, n_components)
        diff = projected - self._mean                      # (N, n_components)
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self._inv_cov, diff))

    def score_patches(
        self,
        train_patches: np.ndarray,  # kept for API consistency — not used
        test_patches: np.ndarray,   # (N_test, n_patches, D)
    ) -> np.ndarray:                # (N_test, n_patches)
        """
        Patch-level scoring. Requires fit() with (N, n_patches, D) input.
        All heavy computation is already done in fit() — this is pure inference.
        """
        if not self._patch_mode:
            raise RuntimeError(
                "Scorer was fitted in CLS mode. "
                "Call fit() with (N, n_patches, D) patch embeddings for patch-level scoring."
            )
        if not hasattr(self, '_inv_covs'):
            raise RuntimeError("Call fit() before score_patches().")

        N_test, n_patches, D = test_patches.shape
        test_proj = test_patches @ self._proj              # (N_test, n_patches, n_components)
        diffs = test_proj - self._means                    # (N_test, n_patches, n_components)
        return np.sqrt(np.einsum("npc,pcd,npd->np", diffs, self._inv_covs, diffs))