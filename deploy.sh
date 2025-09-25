#!/bin/bash
echo "🚀 Desplegando Fiducia Tickets Classifier..."

# Construir imagen
docker build -t fiducia-tickets-classifier .

# Etiquetar para Docker Hub (opcional)
docker tag fiducia-tickets-classifier tu-usuario/fiducia-tickets-classifier:latest

# Ejecutar
docker-compose up -d

echo "✅ Despliegue completado"
echo "🌐 API disponible en: http://localhost:5002"
echo "📊 Health check: http://localhost:5002/health"