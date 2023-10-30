import json
import os

from flask import Flask, render_template, jsonify, redirect, url_for


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

        return jsonify({})

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
                print(index_ids)
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

    return app
