FROM python:3.10-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install build tools and Poetry
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Install all dependencies including dev group
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-interaction --with dev

# Copy application code and test files
COPY ./src ./src
COPY ./rag_docs ./rag_docs

EXPOSE 5000

# Run the Flask app for normal operation
# The original CMD is incorrect for running a module inside a subdirectory
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.main:app_flask"]
