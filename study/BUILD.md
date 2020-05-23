


find `dirname $(dirname $(which python3))` -name "libpython3.*.dylib"
export PYTHON_LIBRARY=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7m/lib/libpython3.7.dylib
export PYTHON_INCLUDE_DIRS=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m/
export PATH=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/
export DYLD_LIBRARY_PATH=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/


mkdir build && cd build
cmake .. -DPY_VERSION=3.7 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DON_INFER=ON -DWITH_MKL=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DWITH_AVX=OFF  -DCMAKE_BUILD_TYPE=Debug




inference build:
export PADDLE_ROOT=/Users/taozj/Dropbox/repos/machine_learning/PADDLE/Paddle/inference_root
cd build
cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT -DCMAKE_BUILD_TYPE=Debug -DWITH_PYTHON=OFF -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_NGRAPH=OFF ..
make -j4
make inference_lib_dist

reference: https://paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/build_and_install_lib_cn.html