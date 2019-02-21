# -*- coding: utf-8 -*-
from flask import Flask, render_template, make_response, jsonify
from flask_sqlalchemy import SQLAlchemy
from functools import wraps, update_wrapper
from datetime import datetime
from score_model import Score
import logging
import random


logger = logging.getLogger(__name__)


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache,' + \
                                            'must-revalidate, post-check=0,' + \
                                            ' pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


# configuration ####
app = Flask(__name__, static_folder='static', static_url_path='')
app.config.from_object('config.DevelopmentConfig')

db = SQLAlchemy(app)

border_colors = []
MAX_COLOR = 100


def r():
    return random.randint(0, 255)


def gen_border_colors():
    for i in range(MAX_COLOR):
        border_colors.append('#%02X%02X%02X' % (r(), r(), r()))


# routes
@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
@nocache
def index():
    return render_template('index.html')


def create_chart_data(data, player_count):

    dataset = {}

    labels = []
    min_x = 0
    round_no = data[0].round_no
    max_x = data[0].round_no
    max_y = 0
    step_size = 10 if round_no > 100 else 1
    resp_step_size = max(1, round((data[0].round_no - 100) / 100))
    counter = 0
    for d in data:
        if dataset.get(d.team_name) is None:
            dataset[d.team_name] = {'data': [], 'label': d.team_name,
                                'borderColor': border_colors[counter],
                                'fill': False
                                }
            counter += 1

        dataset[d.team_name]['data'].append(round(d.overall_score, 4))

    dataset_list = []
    for key in dataset:
        d = sorted(dataset[key]['data'])
        dataset[key]['data'] = []
        l = []
        for i in range(0, round_no, resp_step_size):
            dataset[key]['data'].append({'x': i, 'y': d[i]})
            max_y = max(round(d[i] + 200), max_y)
            if labels is None:
                l.append(i)

        dataset[key]['data'].append({'x': round_no, 'y': d[-1]})
        if labels is None:
            l.append(round_no)

        if labels is None:
            labels = l

        dataset_list.append(dataset[key])

    d = {'type': 'line',
         'data': {

            'datasets': dataset_list,
            'labels': labels
          },
         'options': {
                    'title': {
                      'display': False,
                      'text': 'World population per region (in millions)'
                    }, 'tooltips': {
                      'enabled': False
                    }, 'hover': {'mode': None},
                    'scales': {
                       'xAxes': [{
                                   'type': 'linear',
                                   'display': True,
                                   'ticks': {
                                       'min': min_x,
                                       'max': max_x,
                                       'stepSize': step_size
                                       },
                                }],
                       'yAxes': [{
                                   'type': 'linear',
                                   'display': True,
                                   'ticks': {
                                       'min': 0,
                                       'max': max_y
                                    },
                                }]
                     }
            }}

    return d


@app.route('/api/last_result', methods=['GET'])
def last_result():
    sc = db.session.query(Score).order_by(Score.id.desc()).first()
    print(sc)
    runtime = sc.runtime

    sc_all = db.session.query(Score) \
                       .filter(Score.runtime == runtime)\
                       .order_by(Score.id.desc())

    print(runtime)
    player_count = len(db.session.query(Score)
                       .filter(Score.runtime == runtime)
                       .group_by(Score.team_name).all())

    resp = {}
    resp['runtime'] = runtime
    resp['player_count'] = player_count

    details = []
    ranking = []
    counter = 0
    for sc in sc_all:
        details.append(sc.json)
        if counter < player_count:
            ranking.append([sc.team_name, round(sc.overall_score, 4)])
            counter += 1

    resp['details'] = details
    ranking = sorted(ranking, key=lambda l: l[1], reverse=True)
    resp['ranking'] = ranking
    resp['line_chrat'] = create_chart_data(sc_all, player_count)
    resp['runtime'] = runtime
    resp['round_no'] = sc_all[0].round_no

    return jsonify(resp)


if __name__ == '__main__':
    gen_border_colors()
    app.run(host="0.0.0.0", port=int("8080"))
