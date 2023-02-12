del *.obj

del src\cu\*.obj

del src\cuhelper\*.obj

nvcc -c src\cu\BLAS.cu -o src\cu\BLAS.obj

nvcc -c src\cuhelper\cuhelper.cu -o src\cuhelper\cuhelper.obj

cl.exe /c /EHsc /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include" src\game.cpp src\clcontext.cpp src\cudacontext.cpp src\clhelper\clhelper.cpp

lib.exe game.obj clcontext.obj cudacontext.obj clhelper.obj src\cuhelper\cuhelper.obj src\cu\BLAS.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" OpenCL.lib cudart.lib /OUT:lib\GAME.lib