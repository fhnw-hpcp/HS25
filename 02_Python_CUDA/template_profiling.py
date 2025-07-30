from dataclasses import dataclass, field

import numpy as np
from numba import cuda, float32, config

config.CUDA_ENABLE_PYNVJITLINK = 1

@dataclass
class GPUMatrixProcessor:
  matrixA: np.ndarray = field(init=False)
  matrixB: np.ndarray = field(init=False)
  matrixC: np.ndarray = field(init=False)
  d_matrixA: np.ndarray = field(init=False)
  d_matrixB: np.ndarray = field(init=False)
  d_matrixC: np.ndarray = field(init=False)
  validation: np.ndarray = field(init=False)
  start: cuda.event = field(default_factory=cuda.event)
  end: cuda.event = field(default_factory=cuda.event)
  implementations: dict[str, callable] = field(default_factory=dict)

  def __post_init__(self):
    self.matrixA = np.random.rand(2048, 2048).astype(np.float32)
    self.matrixB = np.random.rand(2048, 2048).astype(np.float32)
    self.matrixC = np.zeros((2048, 2048), dtype=np.float32)
    self.d_matrixA = cuda.to_device(self.matrixA)
    self.d_matrixB = cuda.to_device(self.matrixB)
    self.d_matrixC = cuda.to_device(self.matrixC)
    self.validation = np.dot(self.matrixA, self.matrixB)

  def add_implementation(self, func: callable):
    """Adds a new implementation to the processor."""
    self.implementations[func.__name__] = func

  def run_timed_cuda(self, func: callable, *args):
    """Runs a CUDA function and times its execution using events."""
    self.start.record()
    func(*args)
    self.end.record()
    self.end.synchronize()
    elapsed_time = cuda.event_elapsed_time(self.start, self.end)
    return elapsed_time

  def run_implementation(self, name: str, validate: bool = False, mean_time = True):
    """Multiplies matrixA and matrixB using specified implementation."""
    func = self.implementations.get(name, None)
    if func is None:
      print(f"Implementation '{name}' not found.")
      return
    
    kernel_ms = self.run_timed_cuda(func, self.d_matrixA, self.d_matrixB, self.d_matrixC, 2048)
    if validate:
      self.d_matrixC.copy_to_host(self.matrixC)
      if not np.allclose(self.matrixC, self.validation):
        print(f"Validation of {name} failed: Result does not match expected output.")
    print(f"First execution of {name} took: {kernel_ms:.3f}ms.")
    if mean_time:
      t_mean = np.mean([self.run_timed_cuda(func, self.d_matrixA, self.d_matrixB, self.d_matrixC, 2048) for _ in range(10)])
      print(f"Mean execution time of {name} is: {t_mean:.3f}ms.")



#Our global object to play around with the matrices
matrix_processor = GPUMatrixProcessor()

