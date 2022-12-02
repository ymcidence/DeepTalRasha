import tensorflow as tf
import numpy as np
import talrasha as tr


class GaussianDiffusion:
    """
    Contains utilities for the diffusion model.
    """

    def __init__(self, *, betas, tf_dtype=tf.float32):
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == (timesteps,)

        self.betas = tf.constant(betas, dtype=tf_dtype)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf_dtype)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf_dtype)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf_dtype)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1. - alphas_cumprod), dtype=tf_dtype)
        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1. - alphas_cumprod), dtype=tf_dtype)
        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod), dtype=tf_dtype)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod - 1), dtype=tf_dtype)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf_dtype)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf_dtype)
        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=tf_dtype)
        self.posterior_mean_coef2 = tf.constant(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=tf_dtype)


b = np.linspace(.122, .95, 10)

m1 = tr.model.BasicDiffusion(10, b)
m2 = GaussianDiffusion(betas=b)

print('hehe')
