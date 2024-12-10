#!/bin/bash

./bin/ollama serve &
pid=$!

# Check if Ollama is ready (replace with a more robust check if needed)
while ! nc -z localhost 11434 ; do
    sleep 1
    echo "Waiting for Ollama to start..."
done

echo "Pulling models"
ollama pull gemma2 || { echo "Error pulling gemma2"; exit 1; }
ollama pull mxbai-embed-large || { echo "Error pulling mxbai-embed-large"; exit 1; }

wait $pid
exit $? 