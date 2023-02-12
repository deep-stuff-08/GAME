
nvcc -c ..\..\src\cu\HuffmanCompression\HuffmanCompression.cu -o HuffmanCompression.obj

nvcc -c ..\..\src\cu\kernels.cu -o kernels.obj

nvcc -c Compression.cu -o Compression.obj

nvcc Compression.obj HuffmanCompression.obj kernels.obj -o Compression.exe