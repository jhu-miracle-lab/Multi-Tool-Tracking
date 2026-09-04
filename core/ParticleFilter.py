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
        """Vectorised update: vec_obs_func(particles, **kwargs) -> log_weights (N,)"""
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
        self.normalizeWeights()
        if self._particles.shape[1] % 6 != 0:
            raise ValueError(
                "Particle state must contain one or more 6-DoF pose blocks"
            )

        mean_blocks = []
        for start in range(0, self._particles.shape[1], 6):
            block = self._particles[:, start:start + 6]
            mean_trans = np.dot(self._weights, block[:, :3])
            R_mats = Rotation.from_rotvec(block[:, 3:6]).as_matrix()
            R_avg = np.einsum("n,nij->ij", self._weights, R_mats)
            U, _, Vt = np.linalg.svd(R_avg)
            R_proj = U @ Vt
            if np.linalg.det(R_proj) < 0:
                U[:, -1] *= -1
                R_proj = U @ Vt
            mean_rot = Rotation.from_matrix(R_proj).as_rotvec()
            mean_blocks.append(np.concatenate([mean_trans, mean_rot]))

        return np.concatenate(mean_blocks)
