FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py synset.txt NOTICE LICENSE-ONNX ./
COPY templates templates
USER 10001:10001
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "--timeout", "90", "--worker-tmp-dir", "/tmp", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
