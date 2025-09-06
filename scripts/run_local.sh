#!/bin/bash
set -e

# Ativar virtualenv se existir
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Instalar dependências se não existirem
pip install -r requirements.txt

# Criar diretórios necessários
mkdir -p models results/training results/demo

# Adicionar src ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)/src"

case "$1" in
  train)
    echo "🚀 Iniciando treinamento PPO local..."
    echo "📊 Métricas: results/training/"
    echo "💾 Modelo: models/ppo_torcs.zip"
    python3 src/training/train.py --config src/training/configs/ppo.yaml
    ;;
  play)
    echo "🎮 Rodando agente treinado (demo)..."
    echo "🎥 Vídeos e métricas: results/demo/"
    bash examples/demo_play.sh
    ;;
  web)
    echo "🌐 Iniciando servidor web..."
    echo "💻 Interface disponível em: http://localhost:8000"
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "🔧 Uso: $0 {train|play|web}"
    echo "  train: Treina um novo agente PPO"
    echo "  play:  Roda demonstração com gravação de vídeo"
    echo "  web:   Inicia interface web de monitoramento"
    exit 1
    ;;
esac
