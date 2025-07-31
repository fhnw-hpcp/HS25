from dataclasses import dataclass, field
from collections import defaultdict

import cupy
import numpy as np
from numba import cuda, float32, config

config.CUDA_ENABLE_PYNVJITLINK = 1


@dataclass
class GPUMatrixProcessor:
  data: dict[str, dict[str, np.ndarray]] = field(default_factory=lambda: defaultdict(dict))
  validation: np.ndarray = field(init=False)
  matrix_size: int = 2048
  benchmark: dict[str, dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
  start: cuda.event = field(default_factory=cuda.event)
  end: cuda.event = field(default_factory=cuda.event)
  implementations: dict[str, callable] = field(default_factory=dict)
  implementations_mem_type: dict[str, str] = field(default_factory=dict)

  def __post_init__(self):
    self.create_data(self.matrix_size)

  def create_data(self, size: int):
    """Creates random matrices of given size."""
    self.matrix_size = size
    for matrix in ["A", "B", "C"]:  # Remove previous matrices for proper GC on the GPU
      [self.data[matrix].pop(mem, None) for mem in ["cpu", "cuda", "cupy"]]
    
    self.data["A"]["cpu"] = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
    self.data["B"]["cpu"] = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
    self.data["C"]["cpu"] = np.zeros((self.matrix_size, self.matrix_size), dtype=np.float32)
    for matrix in ["A", "B", "C"]:
      self.data[matrix]["cuda"] = cuda.to_device(self.data[matrix]["cpu"])
      self.data[matrix]["cupy"] = cupy.asarray(self.data[matrix]["cuda"])
    # Create validation data using numpy 
    self.validation = np.dot(self.data["A"]["cpu"], self.data["B"]["cpu"])

  def add_implementation(self, func: callable, mem_type: str):
    """Adds a new implementation to the processor - need to specify which type of memory to use (cpu, cuda, cupy)."""
    if mem_type not in ["cpu", "cuda", "cupy"]:
      raise ValueError("Memory type must be one of: 'cpu', 'cuda', 'cupy'.")
    self.implementations[func.__name__] = func
    self.implementations_mem_type[func.__name__] = mem_type

  def run_timed_cuda(self, func: callable, *args):
    """Runs a CUDA function and times its execution using CUDA events."""
    self.start.record()
    func(*args)
    self.end.record()
    self.end.synchronize()
    return cuda.event_elapsed_time(self.start, self.end)

  def run_timed_cpu(self, func: callable, *args):
    """Runs a CPU function and times its execution using time module."""
    start = time.perf_counter_ns()
    func(*args)
    end = time.perf_counter_ns()
    return (end - start) / 1e6  # Convert to milliseconds

  def run_implementation(self, name: str, validate: bool = False, mean_time = True, silent=False):
    """Multiplies matrixA and matrixB using specified implementation."""
    func = self.implementations.get(name, None)
    if func is None:
      print(f"Implementation '{name}' not found.")
      return
    
    mem = self.implementations_mem_type[name]
    run_function = self.run_timed_cpu if mem == "cpu" else self.run_timed_cuda
    kernel_ms = run_function(func, self.data["A"][mem], self.data["B"][mem], self.data["C"][mem], self.matrix_size)
    if validate:
      if mem in ["cuda", "cupy"]:
        self.data["C"]["cuda"].copy_to_host(self.data["C"]["cpu"])
      if not np.allclose(self.data["C"]["cpu"], self.validation):
        print(f"Validation of {name} failed: Result does not match expected output.")
    if not silent:
      print(f"First execution of {name} took: {kernel_ms:.3f}ms.")
    if mean_time:
      t_mean = np.mean([run_function(func, self.data["A"][mem], self.data["B"][mem], self.data["C"][mem], self.matrix_size) for _ in range(5)])
      self.benchmark[name][self.matrix_size] = t_mean
      if not silent:
          print(f"Mean execution time of {name} is: {t_mean:.3f}ms.")




#Our global object to play around with the matrices
matrix_processor = GPUMatrixProcessor()

