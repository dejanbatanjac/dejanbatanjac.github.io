---
published: false
layout: post
title: PyTorch memory tricks
permalink: /pytorch-memory
---
## How to get GPU memory status in PyTorch?

There are few things you can do with **torch.cuda**...

```python
import torch
t = torch.cuda.get_device_properties(0).total_memory
c = torch.cuda.memory_cached(0)
a = torch.cuda.memory_allocated(0)
f = c-a  # free inside cache
print(t,c,a,f)
```

This will provide total memory info (t). **t** corresponds to a max memory on a card.
Then cached memory is a reserved memory that may be or may not be in use.

From that cached memory if some tensor h/b allocated the memory will be in use, else it will be cached but not in use.

## What we can do to free GPU memory

As stated in the previous section, what we may do is to clear cached memory.

It is very simple to do that.




## How to free GPU memory for a specific tensor?




Both obj = None or del obj are similar, except the del will remove the reference.

However, you need to call gc.collect() to free Python memory without restarting the notebook.

If you would like to clear the obj from PyTorch cache also run:

torch.cuda.empty_cache()

After the last command Nvidea smi or nvtop will notice your did.

## Nvidia


key | exp |
---------|----------|
 timestamp | The timestamp of where the query was made in format "YYYY/MM/DD HH:MM:SS.msec".| 
 name |The official product name of the GPU. This is an alphanumeric string. For all products. |
pci.bus_id | PCI bus id as "domain:bus:device.function", in hex.
driver_version|The version of the installed NVIDIA display driver. This is an alphanumeric string.|
pstate| The current performance state for the GPU. States range from P0 (maximum performance) to P12 (minimum performance).|
pcie.link.gen.max | The maximum PCI-E link generation possible with this GPU and system configuration. For example, if the GPU supports a higher PCIe generation than the system supports then this reports the system PCIe generation.|
pcie.link.gen.current|The current PCI-E link generation. These may be reduced when the GPU is not in use.
temperature.gpu|Core GPU temperature. in degrees C.
utilization.gpu | Percent of time over the past sample period during which one or more kernels was executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product. |
utilization.memory | Percent of time over the past sample period during which global (device) memory was being read or written. The sample period may be between 1 second and 1/6 second depending on the product.|
memory.total | Total installed GPU memory.|
memory.free | Total free memory.|
memory.used | Total memory allocated by active contexts. |


