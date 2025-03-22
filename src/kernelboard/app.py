from flask import Flask, render_template, abort
import dotenv
import psycopg2
import os
import mmh3
from datetime import datetime, timezone

app = Flask(__name__)

@app.template_filter('to_color')
def to_color(name: str) -> str:
    """Convert name to a color using the murmur3 hash"""
    colors = [
        '#FF6B6B',
        '#4ECDC4',
        '#45B7D1',
        '#96CEB4',
        '#FFEEAD',
        '#D4A5A5',
        '#9B5DE5',
        '#F15BB5',
        '#00BBF9',
        '#00F5D4',
    ]
    hash = abs(mmh3.hash(name))
    return colors[hash % len(colors)]

@app.template_filter('to_time_left')
def to_time_left(deadline: str | datetime) -> str | None:
    """
    Calculate time left until deadline.

    Returns:
        - formatted string if deadline is in the future
        - None if deadline has passed
    """
    if isinstance(deadline, str):
        try:
            d = datetime.fromisoformat(deadline)
        except ValueError:
            return None
    else:
        d = deadline

    now = datetime.now(timezone.utc)

    if d <= now:
        return None
        
    delta = d - now
    days = delta.days
    hours = delta.seconds // 3600
    return f"{days} days {hours} hours"

@app.template_filter('format_datetime')
def format_datetime(dt: datetime | str) -> str:
    """
    Common formatting for datetime objects.
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    return dt.strftime('%Y-%m-%d %H:%M UTC')

@app.template_filter('format_score')
def format_score(score: float) -> str:
    """Format score as a string with 3 decimal places."""
    return f"{score * 1_000_000:.3f}μs"

# TODO: This is confusing. It's used in index.html. It's not clear what is in
#       the map parameter. It mixes display login into app.py.
@app.template_filter('add_medals')
def add_medals(users: list[dict[str, str | float]]) -> list[tuple[str, str]]:
    """Add medal emojis to first 3 users, returning tuples of (medal+name, formatted_score)."""
    medals = ["🥇", "🥈", "🥉"]

    results = [
        (
            f"{medals[i]}{user['user_name']}",
            format_score(user['score'])
        )
        for i, user in enumerate(users[:3])]

    # Pad with empty rows until we have 3 total.
    while len(results) < 3:
        results.append(("", ""))

    return results

# TODO: Another confusing function. It's used in index.html.
@app.template_filter('select_highest_priority_gpu')
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

# Load environment variables
dotenv.load_dotenv()

def get_db_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

@app.route('/')
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

    with get_db_connection().cursor() as cur:
        cur.execute(query)
        leaderboards = [row[0] for row in cur.fetchall()]
    
    return render_template('index.html', 
                         leaderboards=leaderboards,
                         now=datetime.now(timezone.utc))

@app.route('/leaderboard/<int:id>')
def leaderboard(id: int):
    # TODO Replace multiple %s parameters with a single named parameter.
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
            WHERE id = %s
        ),
        
        -- All the different GPU types for this leaderboard.
        gpu_types AS (
            SELECT DISTINCT gpu_type
            FROM leaderboard.gpu_type
            WHERE leaderboard_id = %s
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
            WHERE NOT r.secret AND r.score IS NOT NULL AND r.passed AND s.leaderboard_id = %s
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

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # TODO Pass the leaderboard ID three times as required by the query
            cur.execute(query, (id, id, id))
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

@app.route('/coming-soon')
def coming_soon():
    return render_template('coming_soon.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)