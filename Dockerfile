# 1. Using Bullseye to avoid the repository 404 errors found in Buster
FROM python:3.10-slim-bullseye

# 2. Update and install awscli
RUN apt update -y && apt install awscli -y

WORKDIR /app

COPY . /app

# 3. Consolidating pip commands for better layer optimization
RUN pip install --no-cache-dir -r requirements.txt

# 4. Streamlining transformers and accelerate installation 
# (Removed the uninstall/reinstall loop to keep the image slim)
RUN pip install --no-cache-dir --upgrade accelerate transformers

CMD ["python3", "app.py"]