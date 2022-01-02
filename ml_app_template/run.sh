#!/usr/bin/bash
PROJECT_PATH="/home/data/cs-dauphine"
LOG_PATH="/home/data/log"
HOSTNAME=$(hostname)
cd $PROJECT_PATH


# lauch services
killall --quiet gunicorn
gunicorn wsgi:app --config=./gunicorn.conf.py &
if [[ $HOSTNAME != *"-prod"* ]]; then
    killall --quiet jupyter-notebook
    python3 -m jupyter notebook --config=./jupyter.conf.py ./notebooks 1> "$LOG_PATH/notebook.log" 2> "$LOG_PATH/notebook_error.log" &
fi
