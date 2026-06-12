import mlx.core as mx


def custom_kernel(data):
    A, B = data
    return A + B
