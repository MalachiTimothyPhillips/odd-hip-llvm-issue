#!/bin/bash
hipcc --genco -O3 -g -ffp-contract=fast   --amdgpu-target=gfx90a:sramecc-:xnack- source.cpp -o my-kernel-binary
