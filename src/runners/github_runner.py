import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from kernelrunner.run import run_config

config = json.loads(Path("payload.json").read_text())
Path("payload.json").unlink()

result = asdict(run_config(config))


# ensure valid serialization
def serialize(obj: object):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


Path("result.json").write_text(json.dumps(result, default=serialize))
