/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_io/core/plugins/file_system_plugins.h"

#include "absl/strings/ascii.h"

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  const char* env_value = getenv("TF_USE_MODULAR_FILESYSTEM");
  std::string load_plugin = env_value ? absl::AsciiStrToLower(env_value) : "";

  info->plugin_memory_allocate = tensorflow::io::plugin_memory_allocate;
  info->plugin_memory_free = tensorflow::io::plugin_memory_free;
  info->num_schemes = 7;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      tensorflow::io::plugin_memory_allocate(info->num_schemes *
                                             sizeof(info->ops[0])));
  tensorflow::io::az::ProvideFilesystemSupportFor(&info->ops[0], "az");
  tensorflow::io::http::ProvideFilesystemSupportFor(&info->ops[1], "http");
  // Load plugins only when the environment variable is set
  if (load_plugin == "true" || load_plugin == "1") {
    tensorflow::io::s3::ProvideFilesystemSupportFor(&info->ops[2], "s3");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[3], "hdfs");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[4], "viewfs");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[5], "har");
    tensorflow::io::gs::ProvideFilesystemSupportFor(&info->ops[6], "gs");
  } else {
    tensorflow::io::s3::ProvideFilesystemSupportFor(&info->ops[2], "s3e");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[3], "hdfse");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[4], "viewfse");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[5], "hare");
    tensorflow::io::gs::ProvideFilesystemSupportFor(&info->ops[6], "gse");
  }
}
