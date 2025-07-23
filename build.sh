# 环境
# gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
sudo yum install libxml2-devel sudo yum install epel-release libgsasl libgsasl-devel libuuid-devel -y

# 编译

export TF_HEADER_DIR=$(python3 -c 'import tensorflow as tf, os; print(os.path.join(tf.sysconfig.get_include()))')
export TF_SHARED_LIBRARY_DIR=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
export TF_SHARED_LIBRARY_NAME=$(python3 -c 'import tensorflow as tf; print("libtensorflow_framework.so.2")')
export _TF_HEADER_DIR=$TF_HEADER_DIR

echo "TF_HEADER_DIR=$TF_HEADER_DIR"
echo "TF_SHARED_LIBRARY_DIR=$TF_SHARED_LIBRARY_DIR"

# bazel编译失败了，就重复多执行几次，最终如果还是不成功，看日志里的错误，往前面找第一次红色ERROR
bazel clean
bazel build -s --verbose_failures --experimental_repo_remote_exec --compilation_mode=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow_io/... //tensorflow_io_gcs_filesystem/...
python setup.py bdist_wheel --data bazel-bin


# 安装

pip3 install --force dist/tensorflow_io-*.whl

# 正常结果：['oss', 'file', 'hdfs', 'viewfs', 'gs', 'har', 'az', '', 'http', 'ram', 'https', 's3']

python -W always -c "
import tensorflow as tf
import tensorflow_io as tfio
import logging, os, sys
logging.basicConfig(level=logging.INFO)
print(tf.io.gfile.get_registered_schemes())
"

