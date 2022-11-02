FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV HOME=/root
ENV APP_PATH=$HOME/litepose
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH $APP_PATH
WORKDIR $APP_PATH

# copy
COPY . $APP_PATH/

# apt-get install
RUN apt-get update || apt-get install -y \
    git \  
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl

# pip install
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r $APP_PATH/requirements.txt \
    pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    pip install 'git+https://github.com/Jeff-sjtu/CrowdPose#subdirectory=crowdpose-api/PythonAPI'
    
RUN sh get_weights.sh

CMD ["bash"]
