# Troubleshooting Guide

## Common Build Issues

### Linkage Errors After Adding New Files

**Symptom**: Undefined reference errors for ANCF3243 or ANCF3443 functions during linking.

**Cause**: CMake's cache may not have detected newly added source files.

**Solution**: Delete the build directory and reconfigure:
```bash
cd /path/to/FERIS
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

The CMakeLists.txt now uses `CONFIGURE_DEPENDS` with file globbing to automatically detect new source files, but a clean rebuild is still recommended after pulling updates that add new files.

### Constexpr Errors in CUDA Kernels

**Symptom**: Compilation errors about `constexpr` usage in device code.

**Cause**: CUDA requires the `--expt-relaxed-constexpr` flag for extended constexpr support.

**Solution**: This flag is now automatically set in CMakeLists.txt (line 23). If you're using a custom build configuration, ensure this flag is included:
```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
```

### Missing CUDA Toolkit

**Symptom**: `Failed to find nvcc` during CMake configuration.

**Solution**: 
1. Ensure CUDA Toolkit is installed
2. Add CUDA to your PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
3. Or specify CUDA location:
   ```bash
   cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ..
   ```

### Eigen3 Not Found

**Symptom**: CMake error about missing Eigen3.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# Or specify location manually
cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 ..
```

### Wrong CUDA Architecture

**Symptom**: Runtime errors or poor performance on your GPU.

**Solution**: The default configuration supports compute capabilities 6.0-8.6. For specific GPUs:
```cmake
# Edit CMakeLists.txt line 24 for your GPU
# RTX 4090: set(CMAKE_CUDA_ARCHITECTURES "89")
# RTX 3090: set(CMAKE_CUDA_ARCHITECTURES "86")
# RTX 2080: set(CMAKE_CUDA_ARCHITECTURES "75")
```

### Out of Memory Errors

**Symptom**: CUDA out of memory during runtime.

**Solution**:
1. Use a smaller mesh
2. Reduce batch size or simulation parameters
3. Use a GPU with more memory

## Getting Help

If issues persist:
1. Check this troubleshooting guide
2. Review the error messages carefully
3. Ensure all dependencies are installed correctly
4. Try a clean rebuild (delete build directory)
5. Open an issue on GitHub with:
   - Your system configuration (OS, CUDA version, GPU model)
   - Complete error messages
   - Steps to reproduce
