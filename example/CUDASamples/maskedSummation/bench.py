import subprocess

detector_sizes = [256*256, 512*512, 1860*2048]
scan_sizes = [8*8, 16*16]
num_masks = [m+1 for m in range(4)]

for d in detector_sizes:
    for s in scan_sizes:
        for m in num_masks:
            subprocess.call(["./buildGPU/vectorAdd", str(d), str(s), str(m), "8"])
