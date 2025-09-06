#!/bin/bash
set -e

# Verificar se o modelo existe
if [ ! -f "models/ppo_torcs.zip" ]; then
    echo "Erro: Modelo treinado não encontrado em models/ppo_torcs.zip"
    echo "Execute o treinamento primeiro usando: ./examples/demo_train.sh"
    exit 1
fi

# Carregar o modelo treinado e executar episódios de demonstração
python -m scripts.play_agent \
    --model models/ppo_torcs.zip \
    --episodes 3 \
    --render True
