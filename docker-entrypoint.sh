#!/bin/bash

echo "Container is running!!!"

args="$@"
echo $args

if [[ -z ${args} ]]; 
then
    echo "No arguments provided. Starting Streamlit app..."
    pipenv run streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
elif [[ ${args} == "streamlit" ]];
then
    echo "Starting Streamlit app..."
    pipenv run streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
elif [[ ${args} == "shell" ]];
then
    echo "Starting interactive shell..."
    pipenv shell
elif [[ ${args} == "cli" ]];
then
    echo "Starting CLI version..."
    pipenv run python run_integrated_agent.py
else
    echo "Running Python script: $args"
    pipenv run python $args
fi

#echo "Starting FastAPI server..."
#exec pipenv run uvicorn server:app --host 0.0.0.0 --port 8080