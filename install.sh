#!/bin/sh

ABSOLUTE_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $ABSOLUTE_PATH # CD to script dir

printf "Installing 'ranking-server'...\n"

virtualenv -p python3 venv
source ./venv/Scripts/activate

pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html

deactivate

printf "Done installing 'ranking-server'.\n"
