FROM jupyter/base-notebook

MAINTAINER Yannik Behr <y.behr@gns.cri.nz>

USER root
RUN apt-get update && \
    apt-get install -y \
    git \
    vim \
    && apt-get clean

# Grant NB_USER permission to /usr/local

USER $NB_USER
# Install Python 3 packages
RUN conda install --quiet --yes \
    'pandas=0.20*' \
    'scipy=1.1*' \
    'pyproj=1.9*' \
    'pytz=2018*' \
    'pytables=3.4*' \
    'matplotlib=2.2*' \
    'cython=0.25*' \
    'cartopy=0.15*'
 
RUN pip install -I -U pip && \
    pip install git+https://github.com/yannikbehr/spectroscopy.git && \
    pip install git+https://github.com/verigak/progress.git

COPY *.py /usr/local/bin/
RUN mkdir -p /home/jovyan/results \
    && chmod a+rwx /home/jovyan/results
CMD ["/bin/bash"]
