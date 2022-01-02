#!/usr/bin/bash
HOME_PATH="/home/data"
PROJECT_PATH="/home/data/cs-dauphine"
HOSTNAME=$(hostname)
cd "$PROJECT_PATH"

# install basic requirements
# sudo is required to install python packages and extensions
sudo python3 -m pip install -r ./requirements.txt
if [[ $HOSTNAME != *"-prod"* ]]; then
    sudo python3 -m pip install -r ./requirements-dev.txt
    # if not prod install jupyter and extensions
    sudo python3 -m pip install -r ./requirements-jupyter.txt
    sudo python3 -m jupyter nbextension enable --py --sys-prefix qgrid
    sudo python3 -m jupyter nbextension enable --py --sys-prefix widgetsnbextension
    # patch qgrid extension : 
    sudo sed -i 's!jupyter-widgets/base...base/js/dialog.!jupyter-widgets/base"!g' /usr/local/share/jupyter/nbextensions/qgrid/index.js
fi


# create expected directories
mkdir -p "$HOME_PATH/log"
mkdir -p "$PROJECT_PATH/notebooks/saved_requests"
mkdir -p "$HOME_PATH/knowledge-graphs"

# retrieve environment variables
cp "/home/s3/$HOSTNAME.env" "$PROJECT_PATH/.env"

# add cronjobs to cron
sudo crontab -u ubuntu cronjobs.txt

# set orchestrateur token
python3 scripts/set_orchestrateur_token.py

# download knowledge graphs
python3 scripts/download_knowledge_graphs.py 

# run tests
python3 -m pytest
