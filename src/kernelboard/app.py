from flask import Flask, render_template, abort
import dotenv
import psycopg2
import os
import mmh3
from datetime import datetime, timezone

app = Flask(__name__)

# Add hash filter to Jinja environment
@app.template_filter('hash')
def hash_filter(s):
    """Convert string to a number using Python's built-in hash"""
    return abs(mmh3.hash(str(s)))

@app.template_filter('to_color')
def to_color(name: str) -> str:
    """Convert name to a color using a murmur3 hash"""
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
def to_time_left(deadline: str) -> str | None:
    """
    Calculate time left until deadline.

    Returns:
        - formatted string if deadline is in the future
        - None if deadline has passed
    """
    try:
        d = datetime.fromisoformat(deadline)
    except ValueError:
        return None

    now = datetime.now(timezone.utc)

    if d <= now:
        return None
        
    delta = d - now
    days = delta.days
    hours = delta.seconds // 3600
    return f"{days} days {hours} hours"

@app.template_filter('format_datetime')
def format_datetime(dt: datetime) -> str:
    """
    Common formatting for datetime objects.
    """
    return dt.strftime('%Y-%m-%d %H:%M UTC')

@app.template_filter('decorate_rank')
def decorate_rank(rank: int) -> str:
    """
    Adds a medal emoji to ranks 1, 2, and 3.
    """
    emoji = ""
    if rank == 1:
        emoji = "🥇"
    elif rank == 2:
        emoji = "🥈"
    elif rank == 3:
        emoji = "🥉"

    return f"{rank} {emoji}"

@app.template_filter('add_medals')
def add_medals(users: list[dict[str, str | float]]) -> list[tuple[str, str]]:
    """Add medal emojis to first 3 users, returning tuples of (medal+name, formatted_score)."""
    medals = ["🥇", "🥈", "🥉"]
    results = [(f"{medals[i]}{user['user_name']}", f"{user['score'] * 1_000_000:.2e}μs")
            for i, user in enumerate(users[:3])]

    # Pad with empty rows until we have 3 total.
    while len(results) < 3:
        results.append(("", ""))

    return results

@app.template_filter('select_gpu_type')
def select_gpu_type(top_users_by_gpu: dict) -> tuple[str, list]:
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
    # "id": 339,
    # "name": "conv2d",
    # "deadline": "2025-04-29T17:00:00-07:00",
    # "top_users_by_gpu": {
    #     "L4": [
    #         {
    #             "rank": 1,
    #             "score": 0.123
    #             "user_name": "alice"
    #         }
    #         ],
    #         "T4": ...
    # This query has unfortunately gotten too complex, and should be refactored.
    query = """
        SELECT jsonb_build_object(
            'id', l.id,
            'name', l.name,
            'deadline', l.deadline,
            'top_users_by_gpu', (
                SELECT jsonb_object_agg(runner, user_data)
                FROM (
                    SELECT r.runner,
                        (SELECT jsonb_agg(top.user_data)
                         FROM (
                             WITH best_submissions AS (
                                 SELECT DISTINCT ON (u.user_name) 
                                     jsonb_build_object(
                                         'user_name', u.user_name,
                                         'score', r2.score,
                                         'rank', RANK() OVER (ORDER BY r2.score ASC)
                                     ) as user_data
                                 FROM leaderboard.runs r2
                                 JOIN leaderboard.submission s ON r2.submission_id = s.id
                                 LEFT JOIN leaderboard.user_info u ON s.user_id = u.id
                                 WHERE s.leaderboard_id = l.id 
                                     AND NOT r2.secret
                                     AND r2.score IS NOT NULL 
                                     AND r2.passed
                                     AND r2.runner = r.runner
                                 ORDER BY u.user_name, r2.score ASC
                             )
                             SELECT user_data
                             FROM best_submissions
                             WHERE (user_data->>'rank')::int <= 3
                             ORDER BY (user_data->>'score')::float ASC
                             LIMIT 3
                         ) top
                        ) as user_data
                    FROM (SELECT DISTINCT runner 
                          FROM leaderboard.runs 
                          WHERE submission_id IN (
                              SELECT id FROM leaderboard.submission 
                              WHERE leaderboard_id = l.id
                          )
                    ) r
                ) gpu_data
            )
        )
        FROM leaderboard.leaderboard l
        WHERE l.deadline > NOW()
        ORDER BY l.deadline ASC;
    """
    with get_db_connection().cursor() as cur:
        cur.execute(query)
        leaderboards = [row[0] for row in cur.fetchall()]
    
    return render_template('index.html', 
                         leaderboards=leaderboards,
                         now=datetime.now(timezone.utc))

def get_rankings(id: int, gpu_type: str, conn: psycopg2.connect):
    """
    For the given leaderboard id and GPU type, get the submissions ordered by
    score.

    Returns:
        Tuple of (file_name, user_name, submission_time, score, rank)
    """

    query = """
        WITH best_submissions AS (
            SELECT DISTINCT ON (u.user_name) s.file_name, u.user_name,
                s.submission_time, r.score
            FROM leaderboard.runs r
            JOIN leaderboard.submission s ON r.submission_id = s.id
            JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
            LEFT JOIN leaderboard.user_info u ON s.user_id = u.id
            WHERE l.id = %s AND r.runner = %s AND NOT r.secret
                    AND r.score IS NOT NULL AND r.passed
            ORDER BY u.user_name, r.score ASC
        )
        SELECT file_name, user_name, submission_time, score,
            RANK() OVER (ORDER BY score ASC) AS rank
        FROM best_submissions
        ORDER BY score ASC;
    """

    result = []

    with conn.cursor() as cur:
        cur.execute(query, (id, gpu_type))
        rankings = cur.fetchall()

        for ranking in rankings:
            file_name = ranking[0]
            user_name = ranking[1]
            submission_time = ranking[2]
            score = ranking[3]
            rank = ranking[4]

            result.append((file_name, user_name, submission_time, score, rank))
                
    return result

@app.route('/leaderboard/<int:id>')
def leaderboard(id: int):
    # Query the database for the specific leaderboard
    query = """
        SELECT name, deadline, 
            task->>'lang' as lang,
            task->>'description' as description,
            task->'files'->>'reference.py' as reference,
            ARRAY_AGG(gt.gpu_type) as gpu_types
        FROM leaderboard.leaderboard l
        LEFT JOIN leaderboard.gpu_type gt ON gt.leaderboard_id = l.id
        WHERE l.id = %s
        GROUP BY l.id, l.name, l.deadline, l.task
    """

    gpu_types = []
    rankings = [] # list of rankings for each gpu type

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (id,))
            result = cur.fetchone()
        
        if result is None:
            abort(404)

        gpu_types = result[5]
        
        for gpu_type in gpu_types:
            rankings.append(get_rankings(id, gpu_type, conn))

    name = result[0]

    deadline = result[1]
    time_left = to_time_left(deadline)

    lang = result[2]
    if lang == 'py':
        lang = 'Python'        

    description = result[3]
    if description is not None:
        description = description.replace('\\n', '\n')

    reference = result[4]
    if reference is not None:
        reference = reference.replace('\\n', '\n')

    gpu_types = result[5]

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