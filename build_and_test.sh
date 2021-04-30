# bazel clean & build

#bazel clean
bazel build -s --verbose_failures //tensorflow_io/... --sandbox_debug > bazel_build.txt 2>&1 | tee
python setup.py bdist_wheel

# install tensorflow io 
pip uninstall tensorflow-io
pip install ./dist/tensorflow_io-0.6.0-cp27-cp27mu-linux_x86_64.whl

# test kafka
pkill -f 'python -m pytest tests/test_kafka_new.py::KafkaDatasetTest::test_kafka_dataset -s'
python -m pytest tests/test_kafka_new.py::KafkaDatasetTest::test_kafka_dataset -s  > test_kafka_new.txt 2>&1 | tee
