# Running a Private Hackathon

This guide covers how to set up and manage a private (invite-only) hackathon using the `allowed_users` feature. Only users on the allowlist can submit to gated leaderboards.

## Prerequisites

- Admin token (`ADMIN_TOKEN` env var or Heroku config)
- API URL (production or local)
- The leaderboard must already exist (created via `update-problems` or `create-leaderboard`)

```bash
# Production
export API_URL="https://discord-cluster-manager-1f6c4782e60a.herokuapp.com"
export TOKEN="<your-admin-token>"

# Local
export API_URL="http://localhost:8000"
export TOKEN="your_secure_token_here"
```

## Day-of Setup

### 1. Collect GitHub usernames

Get a comma-separated list of GitHub usernames from participants. For example:

```
alice, bob, charlie, dave
```

### 2. Set the allowlist on the leaderboard

Convert the comma-separated list into a JSON array and hit the admin endpoint:

```bash
# Single leaderboard
curl -X PUT "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"usernames": ["alice", "bob", "charlie", "dave"]}'
```

If you have multiple leaderboards for the same hackathon, repeat for each:

```bash
for lb in helion-matmul helion-conv helion-attention; do
  curl -X PUT "$API_URL/admin/leaderboards/$lb/allowed-users" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"usernames": ["alice", "bob", "charlie", "dave"]}'
done
```

### 3. Verify the allowlist

```bash
curl -s "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Expected output:

```json
{
    "leaderboard": "helion-matmul",
    "allowed_users": ["alice", "bob", "charlie", "dave"]
}
```

## During the Hackathon

### Add a late participant

Fetch the current list, append the new user, and PUT the updated list:

```bash
# Get current list
CURRENT=$(curl -s "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['allowed_users']))")

# Add new user (e.g. "eve")
UPDATED=$(echo "$CURRENT" | python3 -c "import sys,json; l=json.load(sys.stdin); l.append('eve'); print(json.dumps(l))")

curl -X PUT "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"usernames\": $UPDATED}"
```

Or just PUT the full list again with the new user included — it's a full replacement, not a patch.

### Remove a participant

Same approach — PUT the full list without the removed user.

## After the Hackathon

### Open the leaderboard back up

Set `usernames` to `null` to remove the gate entirely:

```bash
curl -X PUT "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"usernames": null}'
```

## How It Works

- **API submissions (popcorn-cli):** The user's GitHub username (from their OAuth login) is checked against `allowed_users`. If they're not on the list, they get a `403 Forbidden`.
- **Discord submissions:** If `allowed_users` is set, the bot checks whether the Discord user has a role matching the leaderboard name prefix. For example, leaderboard `helion-matmul` requires the `helion` Discord role. Assign this role to participants in your Discord server.
- **NULL = open:** When `allowed_users` is NULL, the leaderboard behaves normally with no restrictions.

## Quick Reference

| Action | Command |
|---|---|
| Set allowlist | `PUT /admin/leaderboards/{name}/allowed-users` with `{"usernames": ["user1", ...]}` |
| View allowlist | `GET /admin/leaderboards/{name}/allowed-users` |
| Clear allowlist (open up) | `PUT /admin/leaderboards/{name}/allowed-users` with `{"usernames": null}` |

## Discord Role Setup

For the Discord bot gate to work, create a role in your Discord server matching the leaderboard name prefix (e.g. `helion` for `helion-*` leaderboards) and assign it to participants. This is a secondary gate — the primary gate is the API-level `allowed_users` check.
