FROM ubuntu:latest@sha256:a0d9e826ab87bd665cfc640598a871b748b4b70a01a4f3d174d4fb02adad07a9

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get upgrade --yes
RUN apt-get install --yes --no-install-recommends wget python3 python3-dev python3-pip python3-h5py python3-yaml python-is-python3 default-jre default-jdk graphviz cython python3-biopython
RUN wget -q https://sourceforge.net/projects/bbmap/files/BBMap_38.94.tar.gz && tar -xf BBMap_38.94.tar.gz
RUN wget -q https://github.com/hyattpd/Prodigal/releases/download/v2.6.3/prodigal.linux -O /usr/local/bin/prodigal && chmod u+x /usr/local/bin/prodigal
RUN pip3 install pathos sklearn tensorflow==2.6.0 configparser pydot

COPY ./deeplasmid.py /deeplasmid/
COPY ./deeplasmid_utils.py /deeplasmid/
COPY ./deeplasmid_model.py /deeplasmid/
COPY ./models /deeplasmid/models
COPY ./sketches/*.sketch.gz /deeplasmid/sketches/

ENV PYTHONPATH=/deeplasmid:${PYTHONPATH}
ENV PATH=${PATH}:/deeplasmid:/bbmap
ENV JAVA_HOME=/usr

RUN apt-get clean
RUN apt-get autoremove

