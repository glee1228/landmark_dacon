FROM  pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN apt-get update \
    && apt-get -y install \
    apt-utils git vim openssh-server

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN pip install --upgrade pip
RUN pip install setuptools
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR /workspace
ADD . .
ENV PYTHONPATH $PYTHONPATH:/workspace

RUN pip install -r requirements.txt

RUN chmod -R a+w /workspace
RUN /bin/bash
