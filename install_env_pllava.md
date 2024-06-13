# install_env_pllava.md

## 1. Clone base env

```bash
conda create -n pllava --clone clean_pytorch_ffmpeg_build
rm ${CONDA_PREFIX}/lib/libffi.7.so ${CONDA_PREFIX}/lib/libffi.so.7 # Fixes ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6 # Fixes ImportError: ${CONDA_PREFIX}/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by ${CONDA_PREFIX}/lib/python3.12/site-packages/torch/lib/libtorch_python.so)
export IMAGEIO_FFMPEG_EXE=ffmpeg
# export IMAGEIO_FREEIMAGE_LIB=

# ImageIO without ffmpeg binary (use system ffmpeg)
pip install imageio imageio-ffmpeg --no-binary imageio-ffmpeg

# OpenCV with CUDA support and system ffmpeg
cd && git clone --recursive https://github.com/opencv/opencv-python.git && cd opencv-python
git submodule sync
git submodule update --init --recursive
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUBLAS=1 -DWITH_CUDA=ON -DWITH_NVCUVID=ON -DWITH_CUBLAS=1 -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DCUDA_ARCH_BIN=7.0 -DOPENCV_ENABLE_NONFREE=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DOPENCV_EXTRA_MODULES_PATH=${HOME}/opencv-python/opencv_contrib/modules -DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so -DCUDA_nvidia-encode_LIBRARY=/usr/local/cuda/lib64/stubs/libnvidia-encode.so" #-DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs" #-DCUDA_nvidia-encode_LIBRARY="

sudo ln -s ${CONDA_PREFIX}/lib/python3.12/site-packages/numpy/core/include/numpy /usr/include/numpy

export ENABLE_HEADLESS=1
export ENABLE_CONTRIB=1
scp ${HOME}/Downloads/Video_Codec_SDK_12.0.16/{Interface/nvEncodeAPI.h,Lib/linux/stubs/x86_64/libnvcuvid.so,Lib/linux/stubs/x86_64/libnvidia-encode.so} my_server:~ # NOTE: on laptop
sudo mv ~/nvEncodeAPI.h /usr/local/cuda/include
sudo mv ~/{libnvcuvid.so,libnvidia-encode.so} /usr/local/cuda/lib64/stubs
pip wheel . --verbose | tee install_opencv.log
pip install opencv_contrib_python_headless-4.10.0.82-cp312-cp312-linux_x86_64.whl

# PyAV without FFMPEG binary (use system ffmpeg)
pip install av --no-binary av

pip install transformers accelerate safetensors peft
# is imageio already installed?
pip install einops gradio moviepy

# Install decord

cd && git clone --recursive https://github.com/zhanwenchen/decord && cd decord
git submodule sync
git submodule update --init --recursive
mkdir build && cd build

cd python
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
make -j

cd ../python
pip install .

```


## Run

```bash
bash scripts/demo.sh
```
