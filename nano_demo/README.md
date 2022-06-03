# lite-pose-demo

<p align="center">
  <img src="demo.gif" width=600>
</p>

## Installation

* Install Pytorch

  ```shell
  wget https://nvidia.box.com/shared/static/2cqb9jwnncv9iii4u60vgd362x4rnkik.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl
  # It seems that the wget command does not work. You may need to download it manually on your browser.
  sudo apt install python3-pip libopenblas-base libopenmpi-dev 
  pip3 install Cython
  pip3 install numpy torch-1.9.0-cp36-cp36m-linux_aarch64.whl
  sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
  cd ~
  git clone --branch release/0.10 https://github.com/pytorch/vision torchvision
  cd torchvision
  export BUILD_VERSION=0.10.0
  python3 setup.py install --user
  cd ..
  ```

* Install TVM:

  ```shell
  # Upgrade cmake
  # Remove the old version
  cd ~
  sudo apt install openssl libssl-dev
  sudo apt remove cmake
  # Install new one
  wget https://github.com/Kitware/CMake/releases/download/v3.21.0-rc2/cmake-3.21.0-rc2-linux-aarch64.sh
  sudo bash cmake-3.21.0-rc2-linux-aarch64.sh --prefix=/usr --exclude-subdir --skip-license
  
  cd ~
  sudo apt install llvm # install llvm which is required by tvm
  git clone --recursive https://github.com/apache/tvm tvm
  cd tvm
  mkdir build
  cp cmake/config.cmake build/
  cd build
  emacs config.cmake
  #[
  # edit config.cmake to change
  # USE_CUDA OFF -> USE_CUDA ON
  # USE_LLVM OFF -> USE_LLVM ON
  # USE_THRUST OFF -> USE_THRUST ON
  # USE_GRAPH_EXECUTOR_CUDA_GRAPH OFF -> USE_GRAPH_EXECUTOR_CUDA_GRAPH ON
  #]
  export PATH=/usr/local/cuda-10.2/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.2
  cmake ..
  make -j4
  cd ..
  cd python
  python3 setup.py install --user
  ```
  
* Install other dependencies:

  ```
  cd nano_demo
  python3 setup.py install --user
  pip3 install munkres wget yacs
  ```
## Running
  ```shell
  python3 start.py
  ```

