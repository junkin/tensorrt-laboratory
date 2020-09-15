#pragma once

#include <nvrpc/executor.h>
#include <trtlab/core/userspace_threads.h>

namespace demo
{
    using thread_t   = trtlab::userspace_threads;
    using executor_t = nvrpc::Executor;
} // namespace demo
