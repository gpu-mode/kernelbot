BEGIN;

INSERT INTO leaderboard.leaderboard
  (id, name, deadline, task, creator_id, forum_id, secret_seed, description)
VALUES
(
  339,
  'conv2d',
  '2025-06-29 17:00:00-07',
  $TASK$
{ ... valid JSON starting with { and ending with } ... }
$TASK$,
  838132355075014667,
  1343279714277527582,
  1828004782,
  $DESC$
Multiline description goes here…
$DESC$
);

COMMIT;
