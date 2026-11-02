import numpy as np
from scipy.spatial.transform import Rotation


class ParticleFilter:
    def __init__(
        self,
        num_states,
        initialDistributionFunc,
        motionModelFunc,
        obsModelFunc,
        num_particles=200,
        min_num_effective_particles=0.5,
    ):

        self._particles = np.zeros((num_particles, num_states))
        self._weights = np.ones((num_particles)) / float(num_particles)
        self._min_num_effective_particles = min_num_effective_particles

        self._initialDistributionFunc = initialDistributionFunc
        self._motionModelFunc = motionModelFunc
        self._obsModelFunc = obsModelFunc

    def normalizeWeights(self):
        self._weights = self._weights / np.sum(self._weights)

    def initializeFilter(self, **kwargs):
        # Initialize particles here, the kwargs is passed to initialDistributionFunc
        for p_idx, _ in enumerate(self._particles):
            self._particles[p_idx], _ = self._initialDistributionFunc(**kwargs)

    def predictionStep(self, **kwargs):
        # Predict particles here, the kwargs is passed to motionModelFunc
        for p_idx, particle in enumerate(self._particles):
            self._particles[p_idx], _ = self._motionModelFunc(particle, **kwargs)

    def updateStep(self, **kwargs):
        # Collect log-likelihoods for numerical stability
        log_obs = np.zeros(len(self._particles))
        for p_idx, particle in enumerate(self._particles):
            log_obs[p_idx] = self._obsModelFunc(particle, **kwargs)

        # Log-sum-exp: subtract max before exponentiating to avoid underflow
        log_obs -= log_obs.max()
        self._weights = self._weights * np.exp(log_obs)
        self.normalizeWeights()

        # Resample if particles are depleted
        if self._min_num_effective_particles > 1.0 / np.sum(self._weights**2):
            self._particles = self._particles[self.resample(), :]
            self._weights = np.ones(len(self._particles)) / float(len(self._particles))

    def resample(self):
        """
        Each of the N strata [i/N, (i+1)/N) draws its own independent uniform
        sample, unlike systematic resampling which shares a single offset
        across all strata.
        """
        n = len(self._weights)
        positions = (np.arange(n) + np.random.uniform(0.0, 1.0, size=n)) / n
        cumsum = np.cumsum(self._weights)
        cumsum[-1] = 1.0  # guard against floating-point overshoot in searchsorted
        return np.searchsorted(cumsum, positions)

    def predictionStep_vec(self, std):
        """Vectorised prediction: zero-mean Gaussian random walk (Wiener process)."""
        self._particles = self._particles + np.random.normal(0, std, self._particles.shape)

    def updateStep_vec(self, vec_obs_func, resample_col_blocks=None, **kwargs):
        """Vectorised update: vec_obs_func(particles, **kwargs) -> log_weights (N,)

        resample_col_blocks: optional list of (start, end) column ranges to
        include when a resample fires. Columns NOT covered by any block are
        left untouched (no row selection/duplication applied to them) --
        they keep evolving via predictionStep_vec's process noise alone.

        This matters for a multi-block shared state (see
        tracking/combined_tracker.py's CombinedArmTracker, num_states =
        6*n_arms): if one block's own observation term is absent this frame
        (e.g. that arm's detector produced nothing), the joint weight is
        driven entirely by the OTHER block(s)' likelihood -- a criterion
        with zero relevance to the absent block's own state. Resampling
        that block anyway under a foreign criterion causes particle
        impoverishment there (duplicated/discarded rows chosen for reasons
        unrelated to that block's own accuracy), which empirically drifts
        further than simply leaving it unresampled (pure random-walk
        diffusion, same as a standalone single-block filter with no
        observation this frame). Passing None (default) resamples every
        column together, i.e. the original behavior.
        """
        log_obs = vec_obs_func(self._particles, **kwargs)
        log_obs -= log_obs.max()
        self._weights = self._weights * np.exp(log_obs)
        self.normalizeWeights()
        if self._min_num_effective_particles > 1.0 / np.sum(self._weights**2):
            idx = self.resample()
            if resample_col_blocks is None:
                self._particles = self._particles[idx, :]
            else:
                resampled = self._particles[idx, :]
                for start, end in resample_col_blocks:
                    self._particles[:, start:end] = resampled[:, start:end]
            self._weights = np.ones(len(self._particles)) / float(len(self._particles))

    def getMeanParticle(self):
        """Weighted mean particle, generalized over any number of concatenated
        6-DOF SE(3) blocks (state = [dtx,dty,dtz,drx,dry,drz] per block) --
        num_states=6 for a single independently-tracked arm, or 6*n_arms for
        a CombinedArmTracker's shared state (see tracking/combined_tracker.py).
        """
        self.normalizeWeights()
        num_states = self._particles.shape[1]
        assert (
            num_states % 6 == 0
        ), f"getMeanParticle() assumes SE(3) blocks of 6 (trans+rotvec), got num_states={num_states}"

        blocks = []
        for b in range(num_states // 6):
            trans_idx = slice(b * 6, b * 6 + 3)
            rot_idx = slice(b * 6 + 3, b * 6 + 6)

            # Translation: weighted Euclidean mean
            mean_trans = np.dot(self._weights, self._particles[:, trans_idx])

            # Rotation: weighted mean on SO(3) via rotation-matrix averaging + SVD projection.
            R_mats = Rotation.from_rotvec(self._particles[:, rot_idx]).as_matrix()  # (N,3,3)
            R_avg = np.einsum("n,nij->ij", self._weights, R_mats)  # weighted sum
            U, _, Vt = np.linalg.svd(R_avg)
            R_proj = U @ Vt
            if np.linalg.det(R_proj) < 0:  # ensure proper rotation (det=+1)
                U[:, -1] *= -1
                R_proj = U @ Vt
            mean_rot = Rotation.from_matrix(R_proj).as_rotvec()

            blocks.append(mean_trans)
            blocks.append(mean_rot)

        return np.concatenate(blocks)
