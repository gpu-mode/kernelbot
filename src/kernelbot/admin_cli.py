import argparse
import json
import os
from urllib.parse import quote

import requests


def _validate_top10(args: argparse.Namespace) -> int:
    api_url = args.api_url or os.getenv("DISCORD_CLUSTER_MANAGER_API_BASE_URL")
    token = os.getenv("ADMIN_TOKEN")
    if not api_url:
        raise SystemExit(
            "Set DISCORD_CLUSTER_MANAGER_API_BASE_URL or pass --api-url."
        )
    if not token:
        raise SystemExit("Set ADMIN_TOKEN.")

    path = "/admin/application-validations/{}/{}".format(
        quote(args.leaderboard, safe=""),
        quote(args.gpu, safe=""),
    )
    params = {"wait": str(args.wait).lower()}
    if args.all_users:
        params["all_users"] = "true"
    response = requests.post(
        f"{api_url.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=None if args.wait else 30,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernelbot-admin")
    parser.add_argument(
        "--api-url",
        help="Kernelbot API URL (defaults to DISCORD_CLUSTER_MANAGER_API_BASE_URL)",
    )
    commands = parser.add_subparsers(required=True)
    validate = commands.add_parser(
        "validate-top10",
        help="Run application validation for a leaderboard's current top 10",
    )
    validate.add_argument("leaderboard")
    validate.add_argument("gpu")
    validate.add_argument(
        "--wait",
        action="store_true",
        help="Wait for all validation jobs and print their results",
    )
    validate.add_argument(
        "--all-users",
        action="store_true",
        help="Validate every ranked user's best submission instead of only the top 10",
    )
    validate.set_defaults(run=_validate_top10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
