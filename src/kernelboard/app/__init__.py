from flask import Flask
import os
from . import color, db, error, index, leaderboard, score, time


def create_app(test_config=None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.getenv('DATABASE_URL'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)

    app.add_template_filter(color.to_color, 'to_color')
    app.add_template_filter(index.select_highest_priority_gpu, 'select_highest_priority_gpu')
    app.add_template_filter(score.format_score, 'format_score')
    app.add_template_filter(time.to_time_left, 'to_time_left')
    app.add_template_filter(time.format_datetime, 'format_datetime')

    app.register_blueprint(index.blueprint)
    app.add_url_rule('/', endpoint='index')

    app.register_blueprint(leaderboard.blueprint)
    app.add_url_rule('/leaderboard/<int:id>', endpoint='leaderboard')

    app.register_blueprint(error.blueprint)
    app.add_url_rule('/coming-soon', endpoint='coming_soon')

    app.errorhandler(404)(error.page_not_found)
    app.errorhandler(500)(error.server_error)

    return app


app = create_app()