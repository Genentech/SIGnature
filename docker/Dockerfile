FROM fedora:42
LABEL maintainer="Jason Anthony Vander Heiden [vandej27@gene.com]" \
      description="SIGnature"

# Bind points
VOLUME /data
VOLUME /models
VOLUME /workspace
VOLUME /scratch

# Tools
COPY docker/start-notebook.sh /usr/local/bin/start-notebook
COPY . /tmp/SIGnature

# Environment
ENV SCDATA_HOME=/data
ENV SCMODEL_HOME=/models

# Update
RUN dnf -y update && dnf clean all

# Install dependencies
RUN dnf install -y \
    bzip2 \
    cmake \
    igraph \
    gcc-c++ \
    git \
    lz4 \
    pandoc \
    python3 \
    python3-aiohttp \
    python3-asciitree \
    python3-biopython \
    python3-cloudpickle \
    python3-Cython \
    python3-numcodecs \
    python3-dask \
    python3-dask+array \
    python3-devel \
    python3-fasteners \
    python3-GitPython \
    python3-h5py \
    python3-igraph \
    python3-jupyter-client \
    python3-jupyterlab_pygments \
    python3-matplotlib \
    python3-matplotlib-scalebar \
    python3-natsort \
    python3-nbconvert \
    python3-nbsphinx \
    python3-notebook \
    python3-numpy \
    python3-pandas \
    python3-pip \
    python3-pydantic \
    python3-pydata-sphinx-theme \
    python3-PyYAML \
    python3-seaborn \
    python3-setuptools \
    python3-scipy \
    python3-stdlib-list \
    python3-texttable \
    python3-toolz \
    python3-tqdm \
    python3-wrapt \
    python3-zarr \
    sudo \
    tar \
    wget \
    zstd \
    && dnf clean all

# Install python dependencies
RUN pip3 install \
	numba \
	tiledb \
	captum \
	torch \
    scanpy \
    tqdm

# Install SIGnature API
RUN cd /tmp/SIGnature \
    && pip install .

# Entry points
CMD ["start-notebook"]
