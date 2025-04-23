FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && pip install --no-cache-dir -r requirements.txt
# CMD ["streamlit","run" ,"app.py"]
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
