#!/bin/bash
set -e

# Ativar virtualenv se existir
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Instalar dependências se não existirem
pip install -r requirements.txt

case "$1" in
  train)
    echo "Iniciando treinamento PPO local..."
    python training/train.py --config training/configs/ppo.yaml
    ;;
  play)
    echo "Rodando agente treinado (demo)..."
    bash examples/demo_play.sh
    ;;
  web)
    echo "Iniciando servidor web..."
    uvicorn web.app:app --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "Uso: $0 {train|play|web}"
    exit 1
    ;;
esac
