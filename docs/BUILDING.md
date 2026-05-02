# Building FERIS

## Prerequisites

### Required
- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher (with nvcc compiler)
- Eigen3 library (3.3 or higher)
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)

### Ubuntu/Debian Installation
```bash
# Install CMake
sudo apt-get update
sudo apt-get install cmake

# Install Eigen3
sudo apt-get install libeigen3-dev

# Install CUDA Toolkit (if not already installed)
# Follow instructions from: https://developer.nvidia.com/cuda-downloads
```

### Fedora/RHEL Installation
```bash
# Install CMake
sudo dnf install cmake

# Install Eigen3
sudo dnf install eigen3-devel

# Install CUDA Toolkit
# Follow instructions from: https://developer.nvidia.com/cuda-downloads
```

## Building the Project

1. Clone the repository:
```bash
git clone https://github.com/Ruochun/FERIS.git
cd FERIS
```

2. Create a build directory:
```bash
mkdir build
cd build
```

3. Configure with CMake:
```bash
cmake ..
```

If Eigen3 is installed in a non-standard location:
```bash
cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 ..
```

4. Build the project:
```bash
make -j$(nproc)
```

**Note**: If you add new source files or encounter linkage errors after pulling updates, delete the build directory and reconfigure:
```bash
cd ..
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Running the Example

After building, run the example from the build directory:
```bash
./bin/test_feat10_nesterov
```

## CUDA Architecture Support

The default CMakeLists.txt supports CUDA compute capabilities 6.0, 7.0, 7.5, 8.0, and 8.6.
To customize for your specific GPU, modify the `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt.

For example, for an RTX 3090 (Ampere, compute capability 8.6):
```cmake
set(CMAKE_CUDA_ARCHITECTURES "86")
```

## Troubleshooting

### "Failed to find nvcc" Error
Make sure CUDA is installed and nvcc is in your PATH:
```bash
which nvcc
nvcc --version
```

### Eigen3 Not Found
If CMake cannot find Eigen3, specify its location:
```bash
cmake -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 ..
```

### Link Errors with CUDA Libraries
Ensure that CUDA libraries are in your library path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Development

For development, you can enable verbose output:
```bash
make VERBOSE=1
```

To clean and rebuild:
```bash
make clean
make -j$(nproc)
```
