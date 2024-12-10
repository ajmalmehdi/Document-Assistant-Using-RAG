./bin/ollama serve &

pid=$!

sleep 5


echo "Pulling llama3 model"
ollama pull gemma2
ollama pull mxbai-embed-large


wait $pid