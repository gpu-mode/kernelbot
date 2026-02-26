import base64
import json
import os
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from libkernelbot.run_eval import run_config

payload = Path("payload.json").read_text()
Path("payload.json").unlink()
payload = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
config = json.loads(payload)

# For model submissions, the archive is stored as a Git blob (too large for
# workflow dispatch inputs). Download it and inject into the config.
if config.get("archive_blob_sha"):
    import urllib.request

    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    sha = config.pop("archive_blob_sha")

    url = f"https://api.github.com/repos/{repo}/git/blobs/{sha}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        blob_data = json.loads(resp.read())
    config["submission_archive"] = blob_data["content"].replace("\n", "")

result = asdict(run_config(config))


# ensure valid serialization
def serialize(obj: object):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


Path("result.json").write_text(json.dumps(result, default=serialize))
