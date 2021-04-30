# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kafka Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
kafka_ops = _load_library('_kafka_ops.so')

class KafkaDataset(data.Dataset):
  """A Kafka Dataset that consumes the message.
  """

  def __init__(self,
               topics,
               servers="localhost",
               security_protocol="SASL_PLAINTEXT",
               sasl_mechanism="SCRAM-SHA-256",
               sasl_username="alice",
               sasl_password="alice-secret",
               group="",
               client_id="cid",
               eof=False,
               timeout=1000):
    """Create a KafkaReader.

    Args:
      topics: A `tf.string` tensor containing one or more subscriptions,
              in the format of [topic:partition:offset:length],
              by default length is -1 for unlimited.
      servers: A list of bootstrap servers.
      group: The consumer group id.
      eof: If True, the kafka reader will stop on EOF.
      timeout: The timeout value for the Kafka Consumer to wait
               (in millisecond).
    """
    self._topics = tensorflow.convert_to_tensor(
        topics, dtype=dtypes.string, name="topics")
    self._servers = tensorflow.convert_to_tensor(
        servers, dtype=dtypes.string, name="servers")
    self._security_protocol = tensorflow.convert_to_tensor(
        security_protocol, dtype=dtypes.string, name="security_protocol")
    self._sasl_mechanism = tensorflow.convert_to_tensor(
        sasl_mechanism, dtype=dtypes.string, name="sasl_mechanism")
    self._sasl_username = tensorflow.convert_to_tensor(
        sasl_username, dtype=dtypes.string, name="sasl_username")
    self._sasl_password = tensorflow.convert_to_tensor(
        sasl_password, dtype=dtypes.string, name="sasl_password")
    self._group = tensorflow.convert_to_tensor(
        group, dtype=dtypes.string, name="group")
    self._client_id = tensorflow.convert_to_tensor(
        client_id, dtype=dtypes.string, name="client_id")
    self._eof = tensorflow.convert_to_tensor(eof, dtype=dtypes.bool, name="eof")
    self._timeout = tensorflow.convert_to_tensor(
        timeout, dtype=dtypes.int64, name="timeout")
    super(KafkaDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return kafka_ops.kafka_dataset(self._topics, self._servers,
                                   self._security_protocol, self._sasl_mechanism,
                                   self._sasl_username, self._sasl_password,
                                   self._group, self._client_id,
                                   self._eof, self._timeout)

  @property
  def output_classes(self):
    return tensorflow.Tensor

  @property
  def output_shapes(self):
    return tensorflow.TensorShape([])

  @property
  def output_types(self):
    return dtypes.string

def write_kafka(message,
                topic,
                servers="localhost",
                name=None):
  """
  Args:
      message: A `Tensor` of type `string`. 0-D.
      topic: A `tf.string` tensor containing one subscription,
        in the format of topic:partition.
      servers: A list of bootstrap servers.
      name: A name for the operation (optional).
  Returns:
      A `Tensor` of type `string`. 0-D.
  """
  return kafka_ops.write_kafka(
      message=message, topic=topic, servers=servers, name=name)
