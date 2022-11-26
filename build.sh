rm lib/libGAME.so
rm src/cu/BLAS.o
export LD_LIBRARY_PATH="/opt/cuda/lib64"
nvcc --compiler-options '-fPIC' -c src/cu/BLAS.cu -o src/cu/BLAS.o
nvcc --compiler-options '-fPIC' -c src/cuhelper/cuhelper.cu -o src/cuhelper/cuhelper.o
g++ src/game.cpp src/clcontext.cpp src/cudacontext.cpp src/clhelper/clhelper.cpp src/cuhelper/cuhelper.o src/cu/BLAS.o -lOpenCL -lcudart --shared -fPIC -o lib/libGAME.so