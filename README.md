# Using PyTorch on Linux with AMD GPU

Assuming ROCm from AMD has been set up following the instructions, below is how to create a project.

1. Create venv
2. Install PyTorch using offical ROCm distribution. Update as needed.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```
3. Install bitsandbytes (not working)
```
git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -S -DBNB_ROCM_ARCH=gfx1030 .
make
pip install .
```
