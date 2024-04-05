# PostgreSQL Index Analysis

A web application for analyzing database index selection approaches.

Based on our extensible and open-source [index selection evaluation platform](https://github.com/hyrise/index_selection_evaluation).

![alt text](https://github.com/klauck/index_analysis/blob/main/screenshot.png?raw=true)

## Installation and Setup

The application is built on our index selection evaluation platform,
which has to be set up and run first:

1) Initialize the required git submodules

```git submodule update --init --recursive```

2) Install PostgreSQL, e.g., version 15

```sudo apt -y install postgresql-15 postgresql-client-15 postgresql-server-dev-15```

3) Set up the database user
   
```sudo -u postgres createuser -s $(whoami);```

4) Install the PostgreSQL extension HypoPG for hypothetical indexes
    
```
cd index_selection_evaluation/hypopg
make
sudo make install
cd ../..
```


5) Create the virtual Python environments and install the required packages.

```
python -m venv .pyvenv
source .pyvenv/bin/activate

pip install -r index_selection_evaluation/requirements.txt
pip install -r requirements.txt
```

6) Run an index selection evaluation to, then, analyze

```
cd index_selection_evaluation
python -m selection [CONIGURATION.JSON]
```


7) Run the web application

```
flask --app flaskr run --debug
```

## Extensibility

Workloads and results to analyze can be added or edited in the [HTML's dropdown](https://github.com/klauck/index_analysis/blob/main/flaskr/templates/home.html) and [backend's benchmark configuration](https://github.com/klauck/index_analysis/blob/main/flaskr/__init__.py).
