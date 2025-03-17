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
def to_time_left(deadline: datetime) -> str | None:
    """
    Calculate time left until deadline.

    Returns:
        - formatted string if deadline is in the future
        - None if deadline has passed
    """
    now = datetime.now(timezone.utc)
    if deadline <= now:
        return None
        
    delta = deadline - now
    days = delta.days
    hours = delta.seconds // 3600
    return f"{days} days {hours} hours"

# Load environment variables
dotenv.load_dotenv()

def get_db_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

@app.route('/')
def index():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT id, name, deadline
                    FROM leaderboard.leaderboard
                    WHERE deadline > NOW()
                    ORDER BY deadline ASC;
                ''')
                leaderboards = cur.fetchall()
        return render_template('index.html', leaderboards=leaderboards)
    except psycopg2.Error as e:
        app.logger.error(f"Database error: {e}")
        return render_template('500.html'), 500

@app.route('/lectures')
def lectures():
    return render_template('lectures.html')

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

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (id,))
            result = cur.fetchone()
            
    if result is None:
        abort(404)

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
                         reference=reference)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)