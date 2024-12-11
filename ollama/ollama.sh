./bin/ollama serve &

pid=$!

sleep 5


echo "Pulling models"
ollama pull gemma2
ollama pull mxbai-embed-large


wait $pid