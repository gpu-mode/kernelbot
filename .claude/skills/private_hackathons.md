# Private Hackathon Whitelist (Quick Ops)

Use `allowed_users` to make a leaderboard opt-in.

## Semantics

- `allowed_users = null` → leaderboard is open to everyone
- `allowed_users = []` → whitelist is enabled but empty (nobody can submit)
- `allowed_users = ["alice", "bob"]` → only those usernames can submit

## Setup

```bash
export API_URL="https://discord-cluster-manager-1f6c4782e60a.herokuapp.com"
export TOKEN="<admin-token>"
```

## Commands

### View current whitelist

```bash
curl -s "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

### Replace full whitelist

```bash
curl -X PUT "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"usernames": ["alice", "bob", "charlie"]}'
```

### Append users

```bash
curl -X POST "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"usernames": ["dave", "eve"]}'
```

### Remove users

```bash
curl -X DELETE "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"usernames": ["eve"]}'
```

### Re-open leaderboard (disable whitelist)

```bash
curl -X PUT "$API_URL/admin/leaderboards/helion-matmul/allowed-users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"usernames": null}'
```
