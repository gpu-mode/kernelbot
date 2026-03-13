WITH ranked AS (
    SELECT
        s.id as submission_id,
        s.leaderboard_id,
        l.name as problem_name,
        s.user_id,
        u.user_name,
        s.code_id,
        s.file_name,
        s.submission_time,
        COALESCE(
            sjs.status,
            CASE
                WHEN s.done AND r.score IS NOT NULL AND r.passed THEN 'succeeded'
                WHEN s.done THEN 'failed'
                ELSE s.status
            END
        ) as status,
        r.score,
        r.passed,
        r.mode,
        r.runner,
        COALESCE(c.old_code, convert_from(c.code, 'UTF8')) as code,
        ROW_NUMBER() OVER (
            PARTITION BY s.leaderboard_id, s.user_id, s.code_id, r.runner
            ORDER BY r.score ASC NULLS LAST, s.submission_time ASC
        ) as rn
    FROM leaderboard.submission s
    JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
    LEFT JOIN leaderboard.user_info u ON s.user_id = u.id
    LEFT JOIN leaderboard.submission_job_status sjs ON s.id = sjs.submission_id
    LEFT JOIN leaderboard.runs r
        ON s.id = r.submission_id AND r.mode = 'leaderboard' AND NOT r.secret
    LEFT JOIN leaderboard.code_files c ON s.code_id = c.id
    WHERE s.leaderboard_id = ANY(%s)
)
SELECT
    submission_id, leaderboard_id, problem_name, user_id, user_name,
    code_id, file_name, submission_time, status, score, passed, mode,
    runner, code
FROM ranked
WHERE rn = 1
ORDER BY problem_name, score ASC NULLS LAST
