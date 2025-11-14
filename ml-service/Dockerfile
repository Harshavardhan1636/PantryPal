FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install bentoml pandas numpy scikit-learn lightgbm

# Copy service code
COPY . .

# Expose BentoML port
EXPOSE 3000

# Start BentoML server
CMD ["bentoml", "serve", "service:svc", "--host", "0.0.0.0", "--port", "3000"]
