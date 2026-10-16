#!/bin/bash
#chmod +x setup.sh

# Aborta o script em caso de qualquer erro
set -e

echo "🚀 Iniciando a configuração do ambiente Verum Vox..."

# 1. Verifica se a pasta .venv já existe, senão cria
if [ ! -d ".venv" ]; then
    echo "📦 Criando ambiente virtual (.venv)..."
    python3 -m venv .venv
else
    echo "ℹ️  Ambiente virtual (.venv) já existe."
fi

# 2. Ativa o ambiente virtual
echo "🔌 Ativando o ambiente virtual..."
source .venv/bin/activate

# 3. Atualiza o pip
echo "⬆️  Atualizando o pip..."
pip install --upgrade pip

# 4. Instala as dependências do requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📥 Instalando dependências do requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  Arquivo requirements.txt não encontrado!"
fi

echo ""
echo "✅ Ambiente configurado com sucesso!"
echo "💡 Para ativar o ambiente no terminal, rode: source .venv/bin/activate"