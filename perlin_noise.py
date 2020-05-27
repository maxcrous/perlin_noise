import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

corners = np.array(((0, 1), (1, 1), (0, 0), (1, 0)))


def smoothstep(order, sample_density):
    """ Defines a smoothstep curve of a specified order.
        See https://en.wikipedia.org/wiki/Smoothstep
    """
    n = order
    x = np.linspace(0, 1, sample_density)
    sum_array = np.zeros(x.shape)

    for k in range(n + 1):
        sum_array += binom(n + k, k) * binom(2 * n + 1, n - k) * (-x) ** k

    S = x ** (n + 1) * sum_array
    return S


def gradient_grid(grid_shape):
    """ Initializes a grid with random gradients. """
    grid_rows, grid_cols = grid_shape
    full_rot = np.radians(360)
    grad_rots = np.random.uniform(0, full_rot, (grid_rows+1, grid_cols+1))
    xs = np.cos(grad_rots)
    ys = np.sin(grad_rots)
    grad_grid = np.stack((xs, ys), axis=-1)
    return grad_grid


def sample_grid(sample_density):
    """ Initializes a sample grid where each pixel
    corresponds with image pixels.
    """
    xs = np.linspace(0.01, 0.99, sample_density)
    ys = xs
    samp_grid = np.stack(np.meshgrid(xs, ys), axis=-1)
    return samp_grid


def distance_grid(samp_grid):
    """ Initializes a grid with vectors pointing from
        corners to sample pixel locations.
    """

    dist_grids = list()
    for corner in corners:
        dist_vecs = samp_grid - corner
        dist_grids.append(dist_vecs)

    return dist_grids


def perlin_2d(grid_shape=(10, 10), sample_density=40):
    """ Returns a 2D Perlin noise image. """

    grad_grid = gradient_grid(grid_shape)
    samp_grid = sample_grid(sample_density)
    dist_grid = distance_grid(samp_grid)

    smooth_weight = smoothstep(2, sample_density)
    x_weights, y_weights = np.meshgrid(smooth_weight, np.flip(smooth_weight))

    pixels = np.zeros((sample_density * grid_shape[0],
                       sample_density * grid_shape[1]))

    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):

            dot_prods = list()
            for idx, corner in enumerate(corners):
                x = corner[0]
                y = corner[1]
                grad = grad_grid[y + row, x + col]
                dist_vectors = dist_grid[idx]
                dot_prod = np.tensordot(dist_vectors, grad, axes=1)
                dot_prods.append(dot_prod)

            a, b, c, d = dot_prods
            ab = a + x_weights * (b - a)
            cd = c + x_weights * (d - c)
            values = ab + y_weights * (cd - ab)
            pixels[row * sample_density: row * sample_density + sample_density,
                   col * sample_density: col * sample_density + sample_density] = values

    return pixels


pixels = perlin_2d()
plt.imshow(pixels, origin='lower', cmap='Greys')
plt.savefig('perlin_noise.png')
