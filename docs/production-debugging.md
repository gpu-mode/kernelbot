# Production Debugging

Use this when a production submission fails but the API logs do not show the full
runner failure. Start with read-only inspection. Do not paste tokens, database
URLs, or runtime environment output into PRs or issue comments.

## Northflank

Install and login:

```bash
npm i -g @northflank/cli
northflank login --name kernelbot-prod --override
northflank context ls
northflank context use --name <context-name>
```

For the current production deployment, use team `kernelbot`, project `deploy`,
and service `bot`:

```bash
northflank list projects --teamId kernelbot --output json --loadAll
northflank list services --teamId kernelbot --project deploy --output json --loadAll
northflank get service --teamId kernelbot --project deploy --service bot --output json
```

Fetch recent runtime logs:

```bash
northflank get service logs \
  --teamId kernelbot \
  --project deploy \
  --service bot \
  --types runtime \
  --lineLimit 1000 \
  --direction backward \
  --output json
```

Filter for submission and runner failures:

```bash
northflank get service logs \
  --teamId kernelbot \
  --project deploy \
  --service bot \
  --types runtime \
  --lineLimit 1000 \
  --direction backward \
  --regexIncludes 'timeout|Timeout|FunctionTimeout|timed_out|Modal|modal|submission|failed'
```

Inspect the deployed commit:

```bash
northflank get service \
  --teamId kernelbot \
  --project deploy \
  --service bot \
  --output json
```

If you need to query the production database, exec into the service so the
internal addon hostname resolves:

```bash
northflank exec service \
  --teamId kernelbot \
  --project deploy \
  --service bot \
  --cmd 'python -u /tmp/read_only_query.py'
```

The database tables commonly needed for submission debugging are:

- `leaderboard.submission`
- `leaderboard.runs`
- `leaderboard.submission_job_status`
- `leaderboard.leaderboard`
- `leaderboard.user_info`

Useful read-only query shape:

```sql
select s.id, s.file_name, u.user_name, lb.name as leaderboard,
       s.mode_category, s.status, s.done, s.submission_time
from leaderboard.submission s
join leaderboard.leaderboard lb on lb.id = s.leaderboard_id
left join leaderboard.user_info u on u.id = s.user_id
where lb.name = '<leaderboard-name>'
order by s.id desc
limit 50;
```

## Modal

The kernel runners are deployed as Modal app `discord-bot-runner`. To inspect
logs locally, install/use Modal and authenticate with either `modal setup` or
the production Modal token from the Northflank service environment:

```bash
uv run modal --version
uv run modal setup
```

Or for a one-off command with token environment variables:

```bash
MODAL_TOKEN_ID=<token-id> \
MODAL_TOKEN_SECRET=<token-secret> \
uv run modal app logs --timestamps discord-bot-runner
```

Look for platform-level cancellations:

```text
Task's current input ... hit its timeout of 300s
[modal-client] Received a cancellation signal while processing input ...
[modal-client] Successfully canceled input ...
```

Those messages mean Modal killed the function before kernelbot's own runner
timeout handling could return a structured `FullResult`.

## Timeout Debugging Checklist

1. Check Northflank `bot` logs for the API request and `Starting Modal run using ...`.
2. Check Modal app logs for platform cancellations or container errors.
3. Query `leaderboard.submission` for `done=false` rows.
4. Query `leaderboard.runs` for partial public/secret runs.
5. Query `leaderboard.submission_job_status` for background job status when the request used the queued path.
6. Confirm whether the Modal app has been redeployed after runner changes; redeploying only Northflank does not update Modal function metadata.
