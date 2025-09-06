#!/bin/bash
set -e

# Criar diretórios necessários
mkdir -p models results/training

# Verificar se está em um ambiente virtual Python
if [ -z "$VIRTUAL_ENV" ] && [ ! -f /.dockerenv ]; then
    echo "❌ Erro: Por favor, ative o ambiente virtual Python primeiro."
    echo "💡 Execute: source .venv/bin/activate"
    exit 1
fi

# Mensagem inicial
echo "🚀 Iniciando treinamento do agente PPO..."
echo "📊 As métricas serão salvas em results/training/"
echo "💾 O modelo será salvo em models/ppo_torcs.zip"

# Executar o treinamento
PYTHONPATH="." python training/train.py --config training/configs/ppo.yaml
