<img src="https://user-images.githubusercontent.com/12446953/208367719-4ef7922f-4001-41f7-aa9f-076e462d1325.png" width="60%">

## Requirements
- Python 3.6/3.7/3.8/3.9
- PyTorch 1.7.1 with CUDA 11.0
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.5.4

## Installation
1. Install Minkowski Engine
```bash
    pip install -r requirements.txt
```

2. Install other requirements from Pip.
```bash
    pip install -r requirements.txt
```

3. Install ``pointnet2`` module.
```bash
    cd pointnet2
    python setup.py install
```

## License Registration
   
Due to the IP issue, currently only the SDK library file of AnyGrasp is available in a licensed manner. Please get the feature id of your machine and fill in the [form](https://forms.gle/XVV3Eip8njTYJEBo6) to apply for the license. After obtaining license follow below instructions

1. Copy `gsnet.*.so` and `lib_cxx.*.so` to this folder according to your Python version (Python>=3.6,<=3.9 is supported). For example, if you use Python 3.6, you can do as follows:
```bash
    cp gsnet_versions/gsnet.cpython-36m-x86_64-linux-gnu.so gsnet.so
    cp ../license_registration/lib_cxx_versions/lib_cxx.cpython-36m-x86_64-linux-gnu.so lib_cxx.so
```

2. Unzip your license and put the folder here as `license`. Refer to [license_registration/README.md](../license_registration/README.md) if you have not applied for license.

3. Put model weights under ``checkpoints/``.

## Demo Code
Run your code like `demo.py` or any desired applications that uses `gsnet.so`. 
```bash
    cd src; sh demo.sh
```
