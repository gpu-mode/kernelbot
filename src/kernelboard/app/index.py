from flask import Blueprint, render_template
from datetime import datetime, timezone
from .db import get_db_connection


blueprint = Blueprint('index', __name__, url_prefix='/')


@blueprint.route('')
def index():
    # Get a list of JSON objects like the following to build the active
    # leaderboard tiles:
    # {
    #     "id": 339,
    #     "name": "conv2d",
    #     "deadline": "2025-04-29T17:00:00-07:00",
    #     "gpu_types": ["L4", "T4"],
    #     "top_users_by_gpu": {
    #         "L4": [
    #             {
    #                 "rank": 1,
    #                 "score": 0.123
    #                 "user_name": "alice"
    #             }, ...
    #         ], ...

    query = """
        WITH

        -- Get basic information about active leaderboards.
        active_leaderboards AS (
            SELECT id, name, deadline FROM leaderboard.leaderboard
            WHERE deadline > NOW()
        ),

        -- Get all the GPU types for each leaderboard.
        gpu_types AS (
            SELECT DISTINCT leaderboard_id, gpu_type FROM leaderboard.gpu_type
            WHERE leaderboard_id IN (SELECT id FROM active_leaderboards)
        ),

        -- Get each user's best run for each GPU type (runner) on the active
        --leaderboards.
        personal_best_candidates AS (
            SELECT r.runner AS runner,
                s.leaderboard_id AS leaderboard_id,
                u.user_name AS user_name,
                r.score AS score,
                RANK() OVER (PARTITION BY s.leaderboard_id, r.runner, u.id
                ORDER BY r.score ASC) AS personal_submission_rank
            FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN active_leaderboards a ON s.leaderboard_id = a.id
                LEFT JOIN leaderboard.user_info u ON s.user_id = u.id
            WHERE NOT r.secret AND r.score IS NOT NULL AND r.passed
        ),

        -- Select only the best run for each user and GPU type.
        personal_best_runs AS (
            SELECT * FROM personal_best_candidates WHERE personal_submission_rank = 1
        ),

        -- Order the personal best runs by score for each leaderboard and GPU type.
        competitive_rankings AS (
            SELECT leaderboard_id, runner, user_name, score,
            RANK() OVER (PARTITION BY leaderboard_id, runner ORDER BY score ASC) AS user_rank
            FROM personal_best_runs)

        -- Build the JSON response.
        SELECT jsonb_build_object(
            'id', l.id,
            'name', l.name,
            'deadline', l.deadline,
            'gpu_types', (SELECT jsonb_agg(gpu_type) FROM gpu_types g WHERE g.leaderboard_id = l.id),
            'top_users_by_gpu',

                -- For each GPU type, get the top 3 users by rank.
                (SELECT jsonb_object_agg(g.gpu_type, (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'rank', r.user_rank,
                            'score', r.score,
                            'user_name', r.user_name
                        )
                    )
                    FROM competitive_rankings r
                    WHERE r.leaderboard_id = l.id
                    AND r.runner = g.gpu_type
                    AND r.user_rank <= 3
                )) FROM gpu_types g)
        )
        FROM active_leaderboards l;
        """

    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(query)
        leaderboards = [row[0] for row in cur.fetchall()]
    
    return render_template('index.html', 
                         leaderboards=leaderboards,
                         now=datetime.now(timezone.utc))


def select_highest_priority_gpu(top_users_by_gpu: dict) -> tuple[str, list]:
    """
    Select the highest priority GPU type that has data.
    Returns tuple of (gpu_type, users) or None if no data available.
    """
    priority = ['H100', 'B200', 'A100', 'L4', 'T4']
    
    for gpu_type in priority:
        if top_users_by_gpu.get(gpu_type):
            return (gpu_type, top_users_by_gpu[gpu_type])
            
    return (None, None)

