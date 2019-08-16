FROM yangcha/caffe-gpu-conda
LABEL maintainer="tanimuitomo tanimutomo@gmail.com"

RUN pip install --upgrade google-api-python-client &&\
    conda install protobuf

WORKDIR /workspace
ADD . /workspace

