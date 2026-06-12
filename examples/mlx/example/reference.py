import mlx.core as mx


ATOL = 1e-3
RTOL = 1e-3


def generate_input(size, seed=42):
    mx.random.seed(seed)
    A = mx.random.normal(shape=(size, size)).astype(mx.float16)
    B = mx.random.normal(shape=(size, size)).astype(mx.float16)
    mx.eval(A, B)
    return A, B


def reference_kernel(data):
    A, B = data
    return A + B


def check_implementation(data, output):
    expected = reference_kernel(data)
    mx.eval(expected)
    if output.shape != expected.shape:
        return f"shape mismatch: expected {expected.shape}, got {output.shape}"
    if not mx.allclose(output, expected, atol=ATOL, rtol=RTOL).item():
        max_diff = mx.max(mx.abs(output - expected)).item()
        return f"mismatch found! max diff: {max_diff}"
    return ""
