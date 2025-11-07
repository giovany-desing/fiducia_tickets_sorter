# Usa una imagen base más completa para construir
FROM python:3.11-slim

# Instala las dependencias del sistema necesarias para compilar paquetes
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer puerto
EXPOSE 5002

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]