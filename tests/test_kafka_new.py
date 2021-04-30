# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for KafkaDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pytest
import sys

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import dtypes          # pylint: disable=wrong-import-position
from tensorflow import errors          # pylint: disable=wrong-import-position
from tensorflow import test            # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data  # pylint: disable=wrong-import-position

import tensorflow_io.kafka as kafka_io # pylint: disable=wrong-import-position

class KafkaDatasetTest(test.TestCase):
  """Tests for KafkaDataset."""

  # The Kafka server has to be setup before the test
  # and tear down after the test manually.
  # The docker engine has to be installed.
  #
  # To setup the Kafka server:
  # $ bash kafka_test.sh start kafka
  #
  # To team down the Kafka server:
  # $ bash kafka_test.sh stop kafka

  def test_kafka_dataset(self):
    print("====Tests for KafkaDataset====")
    """Tests for KafkaDataset."""
    topics = tensorflow.compat.v1.placeholder(dtypes.string, shape=[None])
    servers = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    security_protocol = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    sasl_mechanism = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    sasl_username = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    sasl_password = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    group = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    client_id = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    timeout = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])
    num_epochs = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])
    batch_size = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kafka_io.KafkaDataset(
        topics, 
        servers="localhost:9092",
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",

        group="my_group", 
        sasl_username="my_username",
        sasl_password="my_password",
        client_id="my_username",

        #group="admin_group", 
        #sasl_username="admin",
        #sasl_password="admin-secret",
        #client_id="admin",
        ).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = data.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    init_batch_op = iterator.make_initializer(batch_dataset)
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Test batched and repeated iteration through both files.
      sess.run(
          init_batch_op,
          feed_dict={
              #servers: "localhost",
              #topics: ["my_topic:0:0:4", "my_topic:0:5:-1"],
              topics: ["my_topic:0:-1:-1"], # topic:partition:offset:length
                                                  # FIXME
              #topics: ["test:0:0:4", "test:0:5:-1"],
              #group: "my_group",
              timeout: 10000,
              num_epochs: 10,
              batch_size: 5
          })
      for _ in range(10):
        print("====get_next====")
        #print(get_next.eval(session=sess))
        print("====msg====: ", sess.run(get_next));
        return;
        #self.assertAllEqual([("D" + str(i)).encode() for i in range(5)],
        #                    sess.run(get_next))
        #self.assertAllEqual([("D" + str(i + 5)).encode() for i in range(5)],
        #                    sess.run(get_next))


  @pytest.mark.skipif(
      (hasattr(tensorflow, "version") and
       tensorflow.version.VERSION.startswith("2.0.")), reason=None)
  def test_kafka_dataset_jave_and_restore(self):
    """Tests for KafkaDataset save and restore."""
    g = tensorflow.Graph()
    with g.as_default():
      topics = tensorflow.compat.v1.placeholder(dtypes.string, shape=[None])
      num_epochs = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])

      repeat_dataset = kafka_io.KafkaDataset(
          topics, group="test", eof=True).repeat(num_epochs)
      iterator = repeat_dataset.make_initializable_iterator()
      get_next = iterator.get_next()

      it = tensorflow.data.experimental.make_saveable_from_iterator(iterator)
      g.add_to_collection(tensorflow.GraphKeys.SAVEABLE_OBJECTS, it)
      saver = tensorflow.train.Saver()

      model_file = "/tmp/test-kafka-model"
      with self.cached_session() as sess:
        sess.run(iterator.initializer,
                 feed_dict={topics: ["test:0:0:4"], num_epochs: 1})
        for i in range(3):
          self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))
        # Save current offset which is 2
        saver.save(sess, model_file, global_step=3)

      checkpoint_file = "/tmp/test-kafka-model-3"
      with self.cached_session() as sess:
        saver.restore(sess, checkpoint_file)
        # Restore current offset to 2
        for i in [2, 3]:
          self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))


  def test_write_kafka(self):
    """test_write_kafka"""
    channel = "e{}e".format(time.time())

    # Start with reading test topic, replace `D` with `e(time)e`,
    # and write to test_e(time)e` topic.
    dataset = kafka_io.KafkaDataset(
        topics=["test:0:0:4"], group="test", eof=True)
    dataset = dataset.map(
        lambda x: kafka_io.write_kafka(
            tensorflow.strings.regex_replace(x, "D", channel),
            topic="test_"+channel))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Basic test: read from topic 0.
      sess.run(init_op)
      for i in range(5):
        self.assertEqual((channel + str(i)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    # Reading from `test_e(time)e` we should get the same result
    dataset = kafka_io.KafkaDataset(
        topics=["test_"+channel], group="test", eof=True)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(5):
        self.assertEqual((channel + str(i)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
