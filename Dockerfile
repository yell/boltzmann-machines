FROM python:2

RUN mkdir -p /var/app
WORKDIR /var/app
COPY . /var/app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
