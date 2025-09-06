#!/bin/bash
set -e

VENV_DIR=".venv"
PYTHON="python3"

# Verifica se o Python 3 está instalado
if ! command -v $PYTHON &> /dev/null; then
    echo "Erro: Python 3 não encontrado. Por favor, instale o Python 3."
    exit 1
fi

echo "Criando ambiente virtual em $VENV_DIR..."

# Remove o ambiente virtual existente se houver
if [ -d "$VENV_DIR" ]; then
    echo "Removendo ambiente virtual existente..."
    rm -rf "$VENV_DIR"
fi

# Cria novo ambiente virtual
$PYTHON -m venv $VENV_DIR

# Ativa o ambiente virtual
source "$VENV_DIR/bin/activate"

# Atualiza pip
echo "Atualizando pip..."
pip install --upgrade pip

# Instala as dependências
echo "Instalando dependências..."
pip install -r requirements.txt

echo "
Ambiente virtual criado com sucesso em $VENV_DIR

Para ativar o ambiente virtual:
source $VENV_DIR/bin/activate

Para desativar:
deactivate
"
