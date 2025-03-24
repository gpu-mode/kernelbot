from flask import abort, Blueprint, render_template
from .db import get_db_connection
from .time import to_time_left


blueprint = Blueprint('leaderboard', __name__, url_prefix='/leaderboard')


@blueprint.route('/<int:leaderboard_id>')
def leaderboard(leaderboard_id: int):
    query = """
        WITH
        
        -- Basic info about the leaderboard.
        leaderboard_info AS (
            SELECT
                name,
                deadline,
                task->>'lang' AS lang,
                task->>'description' AS description,
                task->'files'->>'reference.py' AS reference
            FROM leaderboard.leaderboard
            WHERE id = %(leaderboard_id)s
        ),
        
        -- All the different GPU types for this leaderboard.
        gpu_types AS (
            SELECT DISTINCT gpu_type
            FROM leaderboard.gpu_type
            WHERE leaderboard_id = %(leaderboard_id)s
        ),
        
        -- All the runs on this leaderboard. For each user and GPU type, the
        -- user's runs on that GPU type are ranked by score.
        ranked_runs AS (
            SELECT r.runner AS runner,
                u.user_name AS user_name,
                r.score AS score,
                s.submission_time AS submission_time,
                s.file_name AS file_name,
                RANK() OVER (PARTITION BY r.runner, u.id ORDER BY r.score ASC) AS rank
            FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                LEFT JOIN leaderboard.user_info u ON s.user_id = u.id
            WHERE NOT r.secret AND r.score IS NOT NULL AND r.passed AND s.leaderboard_id = %(leaderboard_id)s
        ),
        
        -- From ranked_runs, keep only the top run per user.
        top_runs AS (SELECT * FROM ranked_runs WHERE rank = 1)
        
        SELECT jsonb_build_object(
            'rankings', (SELECT jsonb_object_agg(g.gpu_type, (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'user_name', r.user_name,
                        'score', r.score,
                        'file_name', r.file_name
                    )
                    ORDER BY r.score ASC
                )
                FROM top_runs r WHERE r.runner = g.gpu_type))),
        
            'leaderboard', (SELECT jsonb_build_object(
                'name', name,
                'deadline', deadline,
                'lang', lang,
                'description', description,
                'reference', reference,
                'gpu_types', (SELECT jsonb_agg(gpu_type) FROM gpu_types)
            ) FROM leaderboard_info)
        ) AS result FROM (SELECT gpu_type FROM gpu_types) g;
    """

    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(query, {'leaderboard_id': leaderboard_id})
        result = cur.fetchone()
    
    if result is None or not result[0]:
        abort(404)
        
    # Parse the JSON result
    data = result[0]
        
    # Extract leaderboard info
    leaderboard_data = data['leaderboard']
    name = leaderboard_data['name']
    deadline = leaderboard_data['deadline']
    time_left = to_time_left(deadline)
    
    lang = leaderboard_data['lang']
    if lang == 'py':
        lang = 'Python'
        
    description = leaderboard_data['description']
    if description is not None:
        description = description.replace('\\n', '\n')
        
    reference = leaderboard_data['reference']
    if reference is not None:
        reference = reference.replace('\\n', '\n')
        
    gpu_types = leaderboard_data['gpu_types']
    gpu_types.sort()

    rankings = {}
    for gpu_type, ranking_ in data['rankings'].items():
        ranking = []
        prev_score = None

        for i, entry in enumerate(ranking_):
            entry['rank'] = i + 1 

            if prev_score is not None:
                entry['prev_score'] = entry['score'] - prev_score
            else:
                entry['prev_score'] = None

            ranking.append(entry)

            prev_score = entry['score']

        if len(ranking) > 0:
            rankings[gpu_type] = ranking
    
    return render_template('leaderboard.html',
                         name=name,
                         deadline=deadline,
                         time_left=time_left,
                         lang=lang,
                         gpu_types=gpu_types,
                         description=description,
                         reference=reference,
                         rankings=rankings)
