FROM envoyproxy/envoy:37bfd8ac347955661af695a417492655b21939dc

RUN apt update && apt install -y --no-install-recommends python3-pip vim-tiny

COPY requirements.txt /tmp/requirements.txt


RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r /tmp/requirements.txt

WORKDIR /work

COPY . .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-

ENTRYPOINT ["python3", "run_loadbalancer.py"]
