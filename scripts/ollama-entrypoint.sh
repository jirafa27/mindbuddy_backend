#!/bin/sh
ollama serve &
SERVER_PID=$!
until ollama list 2>/dev/null; do sleep 1; done
ollama list | grep -q 'qwen2.5:14b' || ollama pull qwen2.5:14b
wait $SERVER_PID
