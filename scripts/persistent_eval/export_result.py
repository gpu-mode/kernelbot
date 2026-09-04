"""Export complete evidence without exceeding the provider's log line limit."""

import base64
import gzip
import hashlib
import json


def emit_result(result):
    """Emit compressed chunks and a checksum for reliable local reconstruction."""
    raw = json.dumps(result).encode()
    encoded = base64.b64encode(gzip.compress(raw)).decode()
    for offset in range(0, len(encoded), 24000):
        print("EXPERIMENT_CHUNK=" + encoded[offset : offset + 24000], flush=True)
    print("EXPERIMENT_END=" + hashlib.sha256(raw).hexdigest(), flush=True)
