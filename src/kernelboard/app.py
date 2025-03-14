from flask import Flask, render_template
from datetime import datetime, UTC
import dotenv
import psycopg2
import os
import hashlib

app = Flask(__name__)

# Add hash filter to Jinja environment
@app.template_filter('hash')
def hash_filter(s):
    """Convert string to a number using Python's built-in hash"""
    return abs(hash(str(s)))

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
                    SELECT 
                        name,
                        (
                            SELECT string_agg(gpu_type, ', ' ORDER BY gpu_type)
                            FROM leaderboard.gpu_type
                            WHERE leaderboard_id = l.id
                        ) as gpu_types,
                        CONCAT(
                            EXTRACT(DAY FROM deadline - NOW())::INTEGER, ' days ',
                            EXTRACT(HOUR FROM deadline - NOW())::INTEGER, ' hours'
                        ) as time_left
                    FROM leaderboard.leaderboard l
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

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)