#
# Copyright (C) 2012-2025 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
:file: python/SHE_PPT/she_io/profiling.py

:date: 04/1122
:author: Gordon Gibb

"""

import logging
import os
import psutil
import time
from queue import Queue

logger = logging.getLogger(__name__)
LOG_LEVEL = logging.INFO
PROFILING_QUEUE = Queue()

def io_stats(f=None, prof_queue=None):
    """A decorator that will print some IO statistics for the function to be called"""
    def decorator(func):
        def profile(*args, **kwargs):
            p = psutil.Process(os.getpid())
            c0 = p.io_counters()
            m = p.memory_info()
            read0 = c0.read_count
            bytes0 = c0.read_chars

            t0 = time.time()
            ret = func(*args, **kwargs)
            t1 = time.time()

            c1 = p.io_counters()
            read1 = c1.read_count - 2  # 2 read ops used in getting the read count twice
            bytes1 = c1.read_chars - 97  # 97 bytes used in getting the bytes read

            open_files = len(p.open_files())

            prof = {"read_ops": read1 - read0,
                    "read": (bytes1 - bytes0) // 1024,
                    "open_files": open_files,
                    "rss": m.rss / 1024**2,
                    "vms": m.vms / 1024**2,
                    "walltime": t1 - t0}

            if prof_queue is not None:
                PROFILING_QUEUE.put(prof)
            else:
                logger.info(
                    "IOSTATS: %s: read_ops = %d, read = %d kB, open_files = %d, rss = %i, vms = %i, walltime = %fs",
                    func.__qualname__,
                    prof["read_ops"],
                    prof["read"],
                    prof["open_files"],
                    prof["rss"],
                    prof["vms"],
                    prof["walltime"],
                )
            return ret
        return profile

    if callable(f):  # used as @io_stats
        return decorator(f)
    else:            # used as @io_stats(queue="...")
        return decorator
