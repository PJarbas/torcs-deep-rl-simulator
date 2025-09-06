#!/bin/bash
set -e

# Criar diretório de resultados se não existir
mkdir -p results/demo

# Verificar se o modelo existe
if [ ! -f "models/ppo_torcs.zip" ]; then
    echo "Erro: Modelo treinado não encontrado em models/ppo_torcs.zip"
    echo "Execute o treinamento primeiro usando: ./examples/demo_train.sh"
    exit 1
fi

# Carregar o modelo treinado e executar episódios de demonstração
echo "🎮 Iniciando demonstração do agente treinado..."
echo "🎥 Os vídeos e métricas serão salvos em results/demo/"

export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)/src"

python3 scripts/play_agent.py \
    --model models/ppo_torcs.zip \
    --episodes 3 \
    --render True \
    --record-video \
    --output results/demo
