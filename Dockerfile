FROM python:3.8-slim-bullseye

RUN apt update
RUN apt install -y \ 
build-essential \
git \
curl \
ca-certificates \
wget \
&& rm -rf /var/lib/apt/lists

WORKDIR /workspace/project

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

