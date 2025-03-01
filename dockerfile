# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies (for example, graphviz for rendering diagrams if needed)
RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all code into the container
COPY . /app

# Install Python dependencies
# (It's good practice to separate requirements.txt, but for brevity we install directly)
RUN pip install --no-cache-dir pandas numpy transformers torch openai mlflow streamlit fastapi uvicorn[standard] matplotlib networkx graphviz

# Expose ports: Streamlit (8501), FastAPI (8000), MLflow (5000)
EXPOSE 8501 8000 5000

# Command to run all services (Streamlit, FastAPI, MLflow) concurrently
# In a real deployment, these might be separate containers or managed by an orchestrator.
CMD mlflow ui --host 0.0.0.0 --port 5000 & \
    streamlit run streamlit_app.py --server.port 8501 --server.enableCORS false & \
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
