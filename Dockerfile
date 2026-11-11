FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 DB_PATH=/app/data/papergap.db REPORTS_DIR=/app/data/reports
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY papergap ./papergap
RUN mkdir -p /app/data
CMD ["python", "-m", "papergap"]
