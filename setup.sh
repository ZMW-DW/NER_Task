echo "📦 syncing environment..."
uv sync
source .venv/bin/activate

echo "📥 downloading bert model..."
modelscope download google-bert/bert-base-multilingual-uncased --include "*.json" --local_dir ./bert-base-multilingual-uncased
modelscope download google-bert/bert-base-multilingual-uncased --include "*.txt" --local_dir ./bert-base-multilingual-uncased
modelscope download google-bert/bert-base-multilingual-uncased --include "*.safetensors" --local_dir ./bert-base-multilingual-uncased

echo "✅ done"