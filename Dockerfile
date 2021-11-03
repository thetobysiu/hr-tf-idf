# Here is the build image
FROM python:3.8.1-slim as builder
RUN apt-get update \
&& apt-get install python3-dev antiword -y \
&& apt-get clean
COPY requirements.txt /app/requirements.txt
WORKDIR app
RUN pip install --upgrade pip && pip install --user -r requirements.txt
COPY . /app
# Here is the production image
FROM python:3.8.1-slim as app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/load.py /app/load.py
COPY --from=builder /app/main.py /app/main.py
WORKDIR app
ENV PATH=/root/.local/bin:$PATH
RUN apt-get update \
&& apt-get install build-essential libpoppler-cpp-dev pkg-config -y \
&& apt-get clean \
&& pip install pdftotext==2.1.4 \
&& python load.py && python -m spacy download en_core_web_sm
ENTRYPOINT python main.py
