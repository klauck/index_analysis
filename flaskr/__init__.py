import json
import logging
import os
import sys

sys.path.append(os.path.abspath('index_selection_evaluation'))

from flask import Flask, render_template, jsonify, redirect, url_for
from selection.cost_evaluation import CostEvaluation
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.index import Index
from selection.result_parser import parse_file
from selection.table_generator import TableGenerator
from selection.query_generator import QueryGenerator
from selection.workload import Workload

CONFIG = {
    "benchmark_name": "tpch",
    "scale_factor": 10,
    "queries": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22],
}
logging.basicConfig(level=logging.DEBUG)

ALGORITHMS = [
    'anytime',
    'auto_admin',
    'db2advis',
    'dexter',
    'drop',
    'extend',
    'relaxation'
]

TABLEAU_COLORS = [
    (31, 119, 180),
    (225, 127, 14),
    (44, 160, 44,),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207)
]

POINT_STYLE = [
    'circle',
    'cross',
    'crossRot',
    'dash',
    'line',
    'rect',
    'rectRounded',
    'rectRot',
    'star',
    'triangle'
]


def create_app(test_config=None, instance_relative_config=True):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def hello():
        # return 'Hello, World!'
        return redirect(url_for('home'))

    @app.route("/home")
    def home():
        return render_template("home.html")

    @app.route('/index_calculation/<index_names_string>')
    def index_calculation(index_names_string):
        print(index_names_string, len(index_names_string))
        data = json.loads(index_names_string)
        print(data, len(data))

        # setup database connection Connection
        dbms_class = PostgresDatabaseConnector
        generating_connector = dbms_class(None, autocommit=True)

        # Attention: This might generate the benchmark tables
        table_generator = TableGenerator(
            CONFIG["benchmark_name"], CONFIG["scale_factor"], generating_connector
        )
        database_name = table_generator.database_name()
        db_connector = PostgresDatabaseConnector(database_name)

        query_generator = QueryGenerator(
            CONFIG["benchmark_name"],
            CONFIG["scale_factor"],
            db_connector,
            CONFIG["queries"],
            table_generator.columns,
        )

        columns_per_name = {}
        for column in table_generator.columns:
            columns_per_name[column.name] = column

        workload = Workload(query_generator.queries)
        print(workload)

        indexes = []
        for index_string_list in json.loads(index_names_string):
            column_names = []
            for column_name in index_string_list:
                column_names.append(columns_per_name[column_name])
            indexes.append(Index(column_names))

        cost_evaluation = CostEvaluation(db_connector)
        query_costs = []
        for query in workload.queries:
            costs = cost_evaluation.calculate_cost(Workload([query]), indexes)
            query_costs.append(costs)

        index_sizes = []
        for index in indexes:
            cost_evaluation.estimate_size(index)
            print(index, index.estimated_size)
            index_sizes.append(index.estimated_size / 10**9)

        return jsonify({'index_sizes': index_sizes, 'query_costs': query_costs})

    @app.route('/index_sizes')
    def get_index_sizes():
        index_names, index_sizes, query_cost_information, query_cost_no_index = retrieve_index_sizes()
        print(index_names, index_sizes)
        index_sizes = [size/10**9 for size in index_sizes]
        queries = [query_number for query_number, _, _ in query_cost_information]
        query_costs = [query_cost for _, _, query_cost in query_cost_information]
        print(query_cost_no_index, type(query_cost_no_index))
        print(query_costs, type(query_costs))
        return jsonify({'index_names': index_names, 'index_sizes': index_sizes, 'query_cost_no_index': query_cost_no_index, 'queries': queries, 'query_costs': query_costs, })

    def retrieve_index_sizes(benchmark='tpch', algorithm='cophy', index_width=2, indexes_per_query=1, storage_budget=5*10**9):
        if algorithm == 'cophy':
            solution_file = f'ILP/{benchmark}_cophy__width{index_width}__per_query{indexes_per_query}__query-based_solution.txt'
        else:
            assert 'Unsupported algorithm'

        # parse solution file to get applied index combinations per query
        index_combinations = None
        with open(os.path.dirname(os.path.abspath(__file__)) + f'/../../index_selection_evaluation/{solution_file}') as f:
            result_str = f.read()

            combinations_per_budget = {}
            for sub_result_str in result_str.split("\n\n\n")[1:-1]:
                lines = sub_result_str.split("\n")
                assert len(lines) == 6, f"{lines}"
                storage_budget = int(float(lines[0].split(" = ")[-1]))
                # relative_workload_costs = float(lines[1].split(" = ")[-1])
                # ilp_time = float(lines[2].split(" = ")[-1])
                used_combinations = map(int, lines[5].split())

                combinations_per_budget[storage_budget] = used_combinations

            index_combinations = list(combinations_per_budget[storage_budget])
            print('index combinations: ', index_combinations)

        # parse the input file, which contains what-if data
        index_sizes = {}  # used to store actual information
        query_costs_for_index_combination = {}  # used to store actual information
        queries = None
        with open(os.path.dirname(os.path.abspath(__file__)) + f'/../../index_selection_evaluation/ILP/{benchmark}_{algorithm}__width{index_width}__per_query{indexes_per_query}__query-based_input.json') as f:
            data = json.loads(f.read())
            queries = data['queries']

            index_per_id = {}  # used to speed up the lookup
            for index in data['index_sizes']:
                index_id, estimated_size, column_names = index['index_id'], index['estimated_size'], index[
                    'column_names']
                index_sizes[tuple(column_names)] = estimated_size
                index_per_id[index_id] = tuple(column_names)
            print(index_sizes)
            print(index_per_id)

            combination_per_id = {}  # used to speed up the lookup
            for index_combination in data['index_combinations']:
                combination_id, index_ids = index_combination['combination_id'], index_combination['index_ids']
                index_set = frozenset([index_per_id[index_id] for index_id in index_ids])
                query_costs_for_index_combination[index_set] = {}
                combination_per_id[combination_id] = index_set

            for q_costs in data['query_costs']:
                query_number, combination_id, costs = q_costs['query_number'], q_costs['combination_id'], q_costs[
                    'costs']
                index_set = combination_per_id[combination_id]
                query_costs_for_index_combination[index_set][query_number] = costs

            used_indexes = set()
            print('index combinations: ', index_combinations)
            for combination_id in index_combinations:
                for index in combination_per_id[combination_id]:
                    used_indexes.add(index)
            print("used indexes: ", used_indexes)

            index_names = []
            index_sizes_l = []
            for index in used_indexes:
                index_names.append(index)
                index_sizes_l.append(index_sizes[index])

            query_cost_information = []
            query_cost_no_index = []
            for i, query_number in enumerate(queries):
                index_set = combination_per_id[index_combinations[i]]
                query_cost_information.append((query_number, index_set, query_costs_for_index_combination[index_set][query_number]))
                query_cost_no_index.append(query_costs_for_index_combination[frozenset()][query_number])
            print(query_cost_information)
            print(query_cost_no_index)

        return index_names, index_sizes_l, query_cost_information, query_cost_no_index

    @app.route('/summary')
    def summary():
        no_index_result = parse_file(os.path.dirname(os.path.abspath(__file__)) + f'/../index_selection_evaluation/benchmark_results/results_no_index_tpch_19_queries.csv')
        print(no_index_result)
        no_index_costs = sum(no_index_result[0][2])

        datasets = []
        for i, algorithm in enumerate(['anytime', 'auto_admin', 'db2advis', 'dexter', 'drop', 'extend', 'relaxation']):
            result = parse_file(os.path.dirname(os.path.abspath(__file__)) + f'/../index_selection_evaluation/benchmark_results/results_{algorithm}_tpch_19_queries.csv')

            data = []
            for line in result:
                data.append({'x': line[0] / 1000, 'y': sum(line[2]) / no_index_costs})
            algorithm_result = {
                'label': algorithm,
                'data': data,
                'borderColor': f'rgba({TABLEAU_COLORS[i][0]}, {TABLEAU_COLORS[i][1]}, {TABLEAU_COLORS[i][2]}, 1)',
                'backgroundColor': f'rgba({TABLEAU_COLORS[i][0]}, {TABLEAU_COLORS[i][1]}, {TABLEAU_COLORS[i][2]}, 0.2)',
                'stepped': True,
                'pointStyle': POINT_STYLE[i],
                'radius': 7
            }
            datasets.append(algorithm_result)
        print(datasets)
        return jsonify(datasets)

    def get_query_cost_per_algorithm_and_budget(algorithm, budget):
        result = parse_file(os.path.dirname(os.path.abspath(
            __file__)) + f'/../index_selection_evaluation/benchmark_results/results_{algorithm}_tpch_19_queries.csv')
        query_cost = None
        for line in result:
            if line[0] / 1000 == budget:
                query_cost = line[2]
        assert query_cost is not None
        return query_cost

    def get_color(algorithm):
        i = ALGORITHMS.index(algorithm)
        return TABLEAU_COLORS[i]

    @app.route('/query_cost_per_algorithm/<algorithm_list_string>')
    def query_cost_per_algorithm(algorithm_list_string):
        print(algorithm_list_string)
        algorithms = json.loads(algorithm_list_string)
        print(algorithms)
        costs_per_algorithm = []
        # 'no_index': get_query_cost_per_algorithm_and_budget('no_index', 0)
        for algorithm in algorithms:
            color = get_color(algorithm[0])
            costs_per_algorithm.append({
                'label': f'{algorithm[0]} {algorithm[1]}',
                'data': get_query_cost_per_algorithm_and_budget(algorithm[0], algorithm[1]),
                'borderColor': f'rgba({color[0]}, {color[1]}, {color[2]}, 1)',
                'backgroundColor': f'rgba({color[0]}, {color[1]}, {color[2]}, {algorithm[1] / 15})',
                'borderWidth': 1
            })
        print(costs_per_algorithm)

        return jsonify(costs_per_algorithm)

    return app
