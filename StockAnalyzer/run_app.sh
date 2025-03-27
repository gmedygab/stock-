#!/bin/bash

# Script di avvio per l'applicazione FinVision
echo "Avvio dell'applicazione FinVision..."

# Controlla se ci sono processi Streamlit in esecuzione e li termina
echo "Controllo processi Streamlit esistenti..."
pkill -f "streamlit run app.py" || true

# Attendi un momento per assicurarsi che i processi siano terminati
sleep 2

# Crea directory .streamlit se non esiste
mkdir -p .streamlit

# Crea il file di configurazione di Streamlit
echo "Configurazione di Streamlit..."
cat > .streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = 5000

[browser]
serverAddress = "0.0.0.0"
serverPort = 5000
EOF

# Avvia l'applicazione Streamlit
echo "Avvio del server Streamlit sulla porta 5000..."
streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true