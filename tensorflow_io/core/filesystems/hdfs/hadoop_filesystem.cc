/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <stdlib.h>
#include <string.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#if defined(_MSC_VER)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "hdfs/hdfs.h"
#include "hdfs/FileSystem.h"
#include "hdfs/InputStream.h"
#include "hdfs/OutputStream.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"

namespace tensorflow {
namespace io {
namespace hdfs {

using namespace hdfs;

// 解析 URI 为 namenode 和 path
static void ParseHadoopPath(const std::string& uri, std::string* nn, std::string* path) {
    // 例如 hdfs://namenode:8020/foo/bar
    size_t p = uri.find("://");
    if (p == std::string::npos) {
        *nn = "default";
        *path = uri;
        return;
    }
    size_t slash = uri.find('/', p + 3);
    *nn = (slash == std::string::npos ? uri.substr(p + 3) : uri.substr(p + 3, slash - p - 3));
    *path = (slash == std::string::npos ? "/" : uri.substr(slash));
}

// 简化 TF_Status 构造
static void SetOK(TF_Status* s) { TF_SetStatus(s, TF_OK, ""); }

class HdfsClient {
public:
    HdfsClient(const std::string& nn) {
        Hdfs::Config conf;

        // 设置 fs.defaultFS 为用户指定的 nn 或默认
        if (!nn.empty() && nn != "default") {
            conf.set("fs.defaultFS", nn);  // 这里是 libhdfs3 的关键配置
        }

        fs_.reset(new Hdfs::FileSystem(conf));

        if (!fs_) {
            throw std::runtime_error("HDFS connect failed: " + nn);
        }
    }

    std::shared_ptr<Hdfs::FileSystem> fs() const { return fs_; }

private:
    std::shared_ptr<Hdfs::FileSystem> fs_;
};

/*** 文件读写器 ***/
namespace tf_random_access_file {
struct Impl {
    std::shared_ptr<Hdfs::InputStream> in;
};

void Cleanup(TF_RandomAccessFile* f) {
    delete static_cast<Impl*>(f->plugin_file);
}

int64_t Read(const TF_RandomAccessFile* f, uint64_t offset, size_t n, char* buf, TF_Status* s) {
    auto imp = static_cast<Impl*>(f->plugin_file);
    try {
        imp->in->seek(offset);
        size_t got = imp->in->read(buf, n);
        SetOK(s);
        return got;
    } catch (const std::exception& e) {
        TF_SetStatus(s, TF_OUT_OF_RANGE, e.what());
        return -1;
    }
}
}  // namespace tf_random_access_file

namespace tf_writable_file {
struct Impl {
    std::shared_ptr<Hdfs::OutputStream> out;
};
void Cleanup(TF_WritableFile* f) {
    delete static_cast<Impl*>(f->plugin_file);
}
void Append(const TF_WritableFile* f, const char* buf, size_t n, TF_Status* s) {
    auto imp = static_cast<Impl*>(f->plugin_file);
    try {
        imp->out->append(buf, n);
        SetOK(s);
    } catch (const std::exception& e) {
        TF_SetStatus(s, TF_INTERNAL, e.what());
    }
}
int64_t Tell(const TF_WritableFile* f, TF_Status* s) {
    SetOK(s);
    return static_cast<int64_t>(static_cast<Impl*>(f->plugin_file)->out->tell());
}
void Flush(const TF_WritableFile* f, TF_Status* s) {
    SetOK(s);
}
void Sync(const TF_WritableFile* f, TF_Status* s) {
    SetOK(s);
}
void Close(const TF_WritableFile* f, TF_Status* s) {
    SetOK(s);
}
}  // namespace tf_writable_file

/*** TF_Filesystem 插件 ***/
namespace tf_hdfs_filesystem {
struct Impl {
    std::unique_ptr<HdfsClient> client;
    absl::Mutex mu;
    std::map<std::string, std::shared_ptr<Hdfs::FileSystem>> cache;
};

hdfsFS Connect(Impl* impl, const std::string& uri, TF_Status* s) {
    std::string nn, path;
    ParseHadoopPath(uri, &nn, &path);
    absl::MutexLock l(&impl->mu);
    if (!impl->cache.count(nn)) {
        impl->client = std::unique_ptr<HdfsClient>(new HdfsClient(nn));
        impl->cache[nn] = impl->client->fs();
    }
    SetOK(s);
    return reinterpret_cast<hdfsFS>(impl->cache[nn].get());
}

void Init(TF_Filesystem* fs, TF_Status* s) {
    fs->plugin_filesystem = new Impl();
    SetOK(s);
}
void Cleanup(TF_Filesystem* fs) {
    delete static_cast<Impl*>(fs->plugin_filesystem);
}

void NewRandomAccessFile(const TF_Filesystem* fs, const char* path, TF_RandomAccessFile* f, TF_Status* s) {
    Impl* impl = static_cast<Impl*>(fs->plugin_filesystem);
    auto clientFs = impl->cache.begin()->second;  // std::shared_ptr<Hdfs::FileSystem>

    // new InputStream 对象
    auto in = std::unique_ptr<Hdfs::InputStream>(new Hdfs::InputStream());
    try {
        // 打开文件
        in->open(*clientFs, path, true);
    } catch (const std::exception& e) {
        TF_SetStatus(s, TF_NOT_FOUND, e.what());
        return;
    }

    // 关联到 Impl
    auto imp = new tf_random_access_file::Impl();
    imp->in = std::move(in);  // 如果 Impl::in 是 unique_ptr<InputStream>
    f->plugin_file = imp;
    SetOK(s);
}

void NewWritableFile(const TF_Filesystem* fs, const char* path, TF_WritableFile* f, TF_Status* s) {
    Impl* impl = static_cast<Impl*>(fs->plugin_filesystem);
    auto clientFs = impl->cache.begin()->second;  // std::shared_ptr<Hdfs::FileSystem>

    // new OutputStream 对象，假设你有类似类
    auto out = std::unique_ptr<Hdfs::OutputStream>(new Hdfs::OutputStream());
    try {
        out->open(*clientFs, path);  // 这里需要确认 OutputStream 的 open 方法签名
    } catch (const std::exception& e) {
        TF_SetStatus(s, TF_INTERNAL, e.what());
        return;
    }

    auto imp = new tf_writable_file::Impl();
    imp->out = std::move(out);  // 如果 Impl::out 是 unique_ptr<OutputStream>
    f->plugin_file = imp;
    SetOK(s);
}


// 其余类似可扩展：Appendable, Stat, Delete, Exists, Rename 等略
}  // namespace tf_hdfs_filesystem

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri) {
    TF_SetFilesystemVersionMetadata(ops);
    ops->scheme = strdup(uri);
    ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
        calloc(1, TF_RANDOM_ACCESS_FILE_OPS_SIZE));
    ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
    ops->random_access_file_ops->read = tf_random_access_file::Read;

    ops->writable_file_ops = static_cast<TF_WritableFileOps*>(
        calloc(1, TF_WRITABLE_FILE_OPS_SIZE));
    ops->writable_file_ops->cleanup = tf_writable_file::Cleanup;
    ops->writable_file_ops->append = tf_writable_file::Append;
    ops->writable_file_ops->tell = tf_writable_file::Tell;
    ops->writable_file_ops->flush = tf_writable_file::Flush;
    ops->writable_file_ops->sync = tf_writable_file::Sync;
    ops->writable_file_ops->close = tf_writable_file::Close;

    ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
        calloc(1, TF_FILESYSTEM_OPS_SIZE));
    ops->filesystem_ops->init = tf_hdfs_filesystem::Init;
    ops->filesystem_ops->cleanup = tf_hdfs_filesystem::Cleanup;
    ops->filesystem_ops->new_random_access_file = tf_hdfs_filesystem::NewRandomAccessFile;
    ops->filesystem_ops->new_writable_file = tf_hdfs_filesystem::NewWritableFile;
}
}  // namespace hdfs
}  // namespace io
}  // namespace tensorflow
