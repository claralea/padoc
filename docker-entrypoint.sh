#!/bin/bash

echo "Container is running!!!"

args="$@"
echo $args

if [[ -z ${args} ]]; 
then
    pipenv shell
else
  pipenv run python $args
fi

#echo "Starting FastAPI server..."
#exec pipenv run uvicorn server:app --host 0.0.0.0 --port 8080