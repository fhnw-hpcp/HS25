import numpy as np
import sys
import time

global_size = int(sys.argv[1])
global_array = np.random.randint(low=0, high=10*global_size, size=global_size)
start_time = time.time()
global_max = np.max(global_array)
end_time = time.time()
print(f"Time to find maximum: {end_time - start_time} sec")
print(f"Global maximum: {global_max:}")
