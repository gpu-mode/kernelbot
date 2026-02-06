# Database Setup

The leaderboard persists information in a Postgres database.

## Local Development

Set up Postgres on your machine, then create a database:

```bash
psql -U postgres -c "CREATE DATABASE clusterdev;"
```

## Migrations

We use [Yoyo Migrations](https://ollycope.com/software/yoyo/) to manage tables, indexes, etc.

### Applying Migrations

```bash
yoyo apply src/migrations -d postgresql://user:password@localhost/clusterdev
```

<details>
<summary>Example yoyo apply session</summary>

```
$ yoyo apply . -d postgresql://user:password@localhost/clusterdev

[20241208_01_p3yuR-initial-leaderboard-schema]
Shall I apply this migration? [Ynvdaqjk?]: y

Selected 1 migration:
  [20241208_01_p3yuR-initial-leaderboard-schema]
Apply this migration to postgresql://user:password@localhost/clusterdev [Yn]: y
Save migration configuration to yoyo.ini?
This is saved in plain text and contains your database password.

Answering 'y' means you do not have to specify the migration source or database connection for future runs [yn]: n
```

</details>

Staging and prod environments use `yoyo apply` with a different database URL.

### Creating New Migrations

```bash
yoyo new src/migrations -m "short_description"
```

Edit the generated file. Do not edit existing migration files - they form an immutable changelog, and yoyo will refuse to reapply changes.

## Expand/Migrate/Contract Pattern

We follow an expand/migrate/contract pattern to allow database migrations without downtime.

### Expansive Changes

Changes that cannot break a running application:
- Adding a new nullable column
- Adding a non-null column with a default value
- Adding an index
- Adding a table

### Contractive Changes

Changes that could break a running application:
- Dropping a table or column
- Adding a NOT NULL constraint
- Adding a unique index

### Workflow

1. **Expand**: Add new elements (nullable columns, new tables, etc.)
2. **Migrate**: Move data to new elements; update code to use them
3. **Contract**: Remove obsolete elements (or add constraints after verifying data satisfies them)

All steps can be written using yoyo migrations.
