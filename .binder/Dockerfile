FROM zivgitlab.wwu.io/pymor/pymor/binder:2022-1-0

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID

USER root
RUN rm -rf /pymor && useradd -d /pymor --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER}

ENV USER=${NB_USER} \
    HOME=/pymor

COPY . /pymor/
RUN chown -R ${NB_USER} /pymor \
  && pip install "jupyter-client<7" "dune-xt~=2021.1.2" "dune-gdt~=2021.1.2" "rise"
USER ${NB_USER}

ENTRYPOINT []
WORKDIR /pymor
