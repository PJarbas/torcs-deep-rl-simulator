#!/bin/bash
set -e

# Verificar se está em um ambiente virtual Python
if [ -z "$VIRTUAL_ENV" ] && [ ! -f /.dockerenv ]; then
    echo "Erro: Por favor, ative o ambiente virtual Python primeiro."
    echo "Execute: source venv/bin/activate"
    exit 1
fi

# Executar o treinamento
PYTHONPATH="." python src/training/train.py --config src/training/configs/ppo.yaml
