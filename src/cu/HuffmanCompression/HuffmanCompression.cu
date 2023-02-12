/*
* huffman function implementations
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "..\..\include\huffman.cuh"

#define BLOCK_SIZE 1024
#define DEBUG 1
#define NUM_BYTES 10240
/*
* Sorting the nodes based on the frequency
* The man frequency is represented by the distinct char count
*/
void sortHuffmanTree(int a, int distinctCharacterCount, int combinedHuffmanNodes){
    for(int i = combinedHuffmanNodes; i < distinctCharacterCount - 1 + a; i++){
        for(int j = combinedHuffmanNodes; j < distinctCharacterCount - 1 + a; j++){

            // perform swapping
            if(huffmanTreeNode[j].frequency > huffmanTreeNode[j + 1].frequency){
                struct huffmanNode tempNode = huffmanTreeNode[j];
                huffmanTreeNode[j] = huffmanTreeNode[j + 1];
                huffmanTreeNode[j + 1] = tempNode;
            }

        }
    }
}

/*
* Build the tree from the sorted results
* The tree here is the min heap
*/
void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes){
    huffmanTreeNode[distinctCharacterCount + i].frequency = huffmanTreeNode[combinedHuffmanNodes].frequency + huffmanTreeNode[combinedHuffmanNodes + 1].frequency;
    huffmanTreeNode[distinctCharacterCount + i].left = & huffmanTreeNode[combinedHuffmanNodes];
    huffmanTreeNode[distinctCharacterCount + i].right = & huffmanTreeNode[combinedHuffmanNodes + 1];
    huffmanTreeNode_head = & (huffmanTreeNode[distinctCharacterCount + i]);
}

/*
* Build the dictionary for the huffman tree
* It will store the bit sequence and their respective lengths
*/
void buildHuffmanDictionary(struct huffmanNode * root, unsigned char * bitSequence, unsigned char bitSequenceLength){
    if(root -> left){
        bitSequence[bitSequenceLength] = 0;
        buildHuffmanDictionary(root -> left, bitSequence, bitSequenceLength + 1);
    }

    if(root -> right){
        bitSequence[bitSequenceLength] = 1;
        buildHuffmanDictionary(root -> right, bitSequence, bitSequenceLength + 1);
    }

    // copy the bit sequence and the length to the dictionary
    if(root -> right == NULL && root -> left == NULL){
        huffmanDictionary.bitSequenceLength[root -> letter] = bitSequenceLength;
        if(bitSequenceLength < 192){
            memcpy(huffmanDictionary.bitSequence[root -> letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
        } else {
            memcpy(bitSequenceConstMemory[root -> letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
            memcpy(huffmanDictionary.bitSequence[root -> letter], bitSequence, 191);
            constMemoryFlag = 1;
        }
    }
}

/*
* Generate data offset array
* Case :- Single run, no overflow
*/
void createDataOffsetArray(unsigned int * compressedDataOffset,
                           unsigned char * inputFileData,
                           unsigned int inputFileLength)
{
    compressedDataOffset[0] = 0;
    for(int i = 0; i < inputFileLength; i++){
        compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
    }
    // not a byte & remaining values
    if(compressedDataOffset[inputFileLength] % 8 != 0){
        compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
    }
}

/*
* Generate data offset array
* Case :- Single run, with overflow
* note : calculate compressed data offset - (1048576 is a safe number that will ensure there is no integer overflow in GPU, it should be minimum 8 * number of threads)
*/
void createDataOffsetArray(unsigned int * compressedDataOffset,
                           unsigned char * inputFileData,
                           unsigned int inputFileLength,
                           unsigned int * integerOverflowIndex,
                           unsigned int * bitPaddingFlag,
                           int numBytes)
{
    int j = 0;
    compressedDataOffset[0] = 0;

    for(int i = 0; i < inputFileLength; i++){
        compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];

        if(compressedDataOffset[i + 1] + numBytes < compressedDataOffset[i]){
            integerOverflowIndex[j] = i;

            if(compressedDataOffset[j] % 8 != 0){
                bitPaddingFlag[j] = 1;
                compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }
            j ++;

        }

    }

    if(compressedDataOffset[inputFileLength] % 8 != 0){
        compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
    }
}

/*
* Generate data offset array
* Case :- Multiple run, with no overflow
*/
void createDataOffsetArray(unsigned int * compressedDataOffset,
                           unsigned char * inputFileData,
                           unsigned int inputFileLength,
                           unsigned int * gpuMemoryOverflow,
                           unsigned int * gpuBitPaddingFlag,
                           long unsigned int memoryRequired)
{
    int j = 0;
    gpuMemoryOverflow[0] = 0;
    gpuBitPaddingFlag[0] = 0;
    compressedDataOffset[0] = 0;

    for(int i = 0; i < inputFileLength; i++){
        compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];

        if(compressedDataOffset[i + 1] > memoryRequired){
            gpuMemoryOverflow[j * 2 + 1] = i;
            gpuMemoryOverflow[j * 2 + 2] = i + 1;

            if(compressedDataOffset[i] % 8 != 0){
                gpuBitPaddingFlag[j + 1] = 1;
                compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }

            j ++;
        }
    }

    if(compressedDataOffset[inputFileLength] % 8 != 0){
        compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
    }

    gpuMemoryOverflow[j * 2 + 1] = inputFileLength;
}

/*
* Generate data offset array
* Case :- Multiple run, with overflow
*/
void createDataOffsetArray(unsigned int * compressedDataOffset,
                           unsigned char * inputFileData,
                           unsigned int inputFileLength,
                           unsigned int * integerOverflowIndex,
                           unsigned int * bitPaddingFlag,
                           unsigned int * gpuMemoryOverflowIndex,
                           unsigned int * gpuBitPaddingFlag,
                           int numBytes,
                           long unsigned int memoryRequired)
{
    int j = 0, k = 0;
    compressedDataOffset[0] = 0;

    for(int i = 0; i < inputFileLength; i++){
        compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
        if(j != 0 && (long unsigned int) compressedDataOffset[i + 1] + compressedDataOffset[integerOverflowIndex[j - 1]] > memoryRequired){
            gpuMemoryOverflowIndex[k * 2 + 1] = i;
            gpuMemoryOverflowIndex[k * 2 + 2] = i + 1;

            if(compressedDataOffset[i] %8 != 0){
                gpuBitPaddingFlag[k + 1] = 1;
                compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }

            k ++;
        }
        else if(compressedDataOffset[i + 1] + numBytes < compressedDataOffset[i]){
            integerOverflowIndex[j] = i;

            // if not a byte
            if(compressedDataOffset[i] % 8 != 0){
                bitPaddingFlag[j] = 1;
                compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }

            j ++;
        }
    }

    // remaining values
    if(compressedDataOffset[inputFileLength] % 8 != 0){
        compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
    }

    gpuMemoryOverflowIndex[j * 2 + 1] = inputFileLength;
}

/*Kernel*/


/*
* Main launching function to load the data on the device
*/
void launchCudaHuffmanCompress(unsigned char * inputFileData,
                               unsigned int * compressedDataOffset,
                               unsigned int inputFileLength,
                               int numberOfKernels,
                               unsigned int integerOverflowFlag,
                               long unsigned int memoryRequired)
{
    struct huffmanDictionary * device_huffmanDictionary;
    unsigned char * device_inputFileData, * device_byteCompressedData;
    unsigned int * device_compressedDataOffset;
    unsigned int * gpuBitPaddingFlag, * bitPaddingFlag;
    unsigned int * gpuMemoryOverflowIndex, * integerOverflowIndex;
    size_t memoryFree, memoryTotal;
    cudaError_t error;

    // generating the offset
    // no integer overflow
    if(integerOverflowFlag == 0){
        // single run no overflow
        if(numberOfKernels == 1){
            createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength);
        }

        // multiple run with no overflow [big files]
        else {
            gpuBitPaddingFlag = (unsigned int *) calloc(numberOfKernels, sizeof(unsigned int));
            gpuMemoryOverflowIndex = (unsigned int *) calloc(numberOfKernels * 2, sizeof(unsigned int));
            createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, gpuMemoryOverflowIndex, gpuBitPaddingFlag, memoryRequired);
        }
    }

    // integer overflow
    else {
        // single run overflow
        if(numberOfKernels == 1){
            bitPaddingFlag = (unsigned int *) calloc(numberOfKernels, sizeof(unsigned int));
            integerOverflowIndex = (unsigned int *) calloc(numberOfKernels * 2, sizeof(unsigned int));
            createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength);
        }

        // multiple run overflow
        else {
            gpuBitPaddingFlag = (unsigned int *) calloc(numberOfKernels, sizeof(unsigned int));
            bitPaddingFlag = (unsigned int *) calloc(numberOfKernels, sizeof(unsigned int));
            integerOverflowIndex = (unsigned int *) calloc(numberOfKernels * 2, sizeof(unsigned int));
            gpuMemoryOverflowIndex = (unsigned int *) calloc(numberOfKernels * 2, sizeof(unsigned int));
            createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, integerOverflowIndex, bitPaddingFlag, gpuMemoryOverflowIndex, gpuBitPaddingFlag, NUM_BYTES, memoryRequired);
        }
    }

    // gpu initiation
    {
        // memory allocation
        error = cudaMalloc((void **) & device_inputFileData, inputFileLength * sizeof(unsigned char));
        if(error != cudaSuccess)
            printf("\nError 1 :: %s", cudaGetErrorString(error));

        error = cudaMalloc((void **) & device_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
        if(error != cudaSuccess)
            printf("\nError 2 :: %s", cudaGetErrorString(error));

        error = cudaMalloc((void **) & device_huffmanDictionary, sizeof(huffmanDictionary));
        if(error != cudaSuccess)
            printf("\nError 3 :: %s", cudaGetErrorString(error));

        // memory copy to device
        error = cudaMemcpy(device_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
        if(error != cudaSuccess)
            printf("\nError 4 :: %s", cudaGetErrorString(error));

        error = cudaMemcpy(device_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if(error != cudaSuccess)
            printf("\nError 5 :: %s", cudaGetErrorString(error));

        error = cudaMemcpy(device_huffmanDictionary, & huffmanDictionary, sizeof(huffmanDictionary), cudaMemcpyHostToDevice);
        if(error != cudaSuccess)
            printf("\nError 6 :: %s", cudaGetErrorString(error));

        // constant memory if required
        if(constMemoryFlag == 1){
            error = cudaMemcpyToSymbol(device_bitSequenceConstMemory, bitSequenceConstMemory, 265 * 255 * sizeof(unsigned char));
            if(error != cudaSuccess)
                printf("\nError Constant :: %s", cudaGetErrorString(error));
        }

    }

    // Single run
    if(numberOfKernels == 1){

        // no overflow
        if(integerOverflowFlag == 0){
            error = cudaMalloc((void **) & device_byteCompressedData, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
			if(error != cudaSuccess)
                printf("\nError 7 :: %s", cudaGetErrorString(error));

			// initialize device_byteCompressedData
			error = cudaMemset(device_byteCompressedData, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if(error != cudaSuccess)
                printf("\nError 8 :: %s", cudaGetErrorString(error));

			// debug
			if(1){
				cudaMemGetInfo(&memoryFree, &memoryTotal);
				printf("\nFree Mem: %zu", memoryFree);
			}

			// run kernel
			compress<<<1, BLOCK_SIZE>>>(device_inputFileData, device_compressedDataOffset, device_huffmanDictionary, device_byteCompressedData, inputFileLength, constMemoryFlag);
			cudaError_t error_kernel = cudaGetLastError();
			if(error_kernel != cudaSuccess)
                printf("\nError Kernel 1 :: %s", cudaGetErrorString(error));

			// copy compressed data from GPU to CPU memory
			error = cudaMemcpy(inputFileData, device_inputFileData, ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			if(error != cudaSuccess)
                printf("\nError 9 :: %s", cudaGetErrorString(error));

			// free allocated memory
			cudaFree(device_inputFileData);
			cudaFree(device_compressedDataOffset);
			cudaFree(device_huffmanDictionary);
			cudaFree(device_byteCompressedData);
		}

		// with overflow
		else {
		   // additional variable to store offset data after integer oveflow
			unsigned char * device_byteCompressedDataOverflow;

			// allocate memory to store offset information
			error = cudaMalloc((void **) & device_byteCompressedData, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("Error 10 :: %s\n", cudaGetErrorString(error));

			error = cudaMalloc((void **) & device_byteCompressedDataOverflow, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("Error 11 :: %s\n", cudaGetErrorString(error));

			// initialize offset data
			error = cudaMemset(device_byteCompressedData, 0, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("Error 12 :: %s\n", cudaGetErrorString(error));

			error = cudaMemset(device_byteCompressedDataOverflow, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("Error 13 :: %s\n", cudaGetErrorString(error));

			// debug
			if(1){
				cudaMemGetInfo(& memoryFree, & memoryTotal);
				printf("Free Mem :: %zu\n", memoryFree);
			}

			// launch kernel
			compress<<<1, BLOCK_SIZE>>>(device_inputFileData, device_compressedDataOffset, device_huffmanDictionary, device_byteCompressedData, device_byteCompressedDataOverflow, inputFileLength, constMemoryFlag, integerOverflowIndex[0]);

			// check status
			cudaError_t error_kernel = cudaGetLastError();
			if (error_kernel != cudaSuccess)
				printf("\nError Kernel 2: %s", cudaGetErrorString(error_kernel));

			// get output data
			if(bitPaddingFlag[0] == 0){
				error = cudaMemcpy(inputFileData, device_inputFileData, (compressedDataOffset[integerOverflowIndex[0]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("Error 14 :: %s\n", cudaGetErrorString(error));

				error = cudaMemcpy(& inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8)], & device_inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8)], ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("Error 15 :: %s\n", cudaGetErrorString(error));
			}
			else{
				error = cudaMemcpy(inputFileData, device_inputFileData, (compressedDataOffset[integerOverflowIndex[0]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("Error 16 :: %s\n", cudaGetErrorString(error));

				unsigned char temp_compByte = inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1];

				error = cudaMemcpy(& inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1], & device_inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8)], ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("Error 17 :: %s\n", cudaGetErrorString(error));

				inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1] = temp_compByte | inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1];
			}

			// free allocated memory
			cudaFree(device_inputFileData);
			cudaFree(device_compressedDataOffset);
			cudaFree(device_huffmanDictionary);
			cudaFree(device_byteCompressedData);
			cudaFree(device_byteCompressedDataOverflow);
		}
    }

    // multiple run
    else{

        // no overflow
		if(integerOverflowFlag == 0){
			error = cudaMalloc((void **) & device_byteCompressedData, (compressedDataOffset[gpuMemoryOverflowIndex[1]]) * sizeof(unsigned char));
			if (error!= cudaSuccess)
				printf("Error 18 :: %s\n", cudaGetErrorString(error));

			// debug
			if(1){
				cudaMemGetInfo(& memoryFree, & memoryTotal);
				printf("\nFree Mem: %zu\n", memoryFree);
			}

			unsigned int pos = 0;
			for(unsigned int i = 0; i < numberOfKernels; i++){
				// initialize d_byteCompressedData
				error = cudaMemset(device_byteCompressedData, 0, compressedDataOffset[gpuMemoryOverflowIndex[1]] * sizeof(unsigned char));
				if (error!= cudaSuccess)
				    printf("Error 19 :: %s\n", cudaGetErrorString(error));

				compress<<<1, BLOCK_SIZE>>>(device_inputFileData, device_compressedDataOffset, device_huffmanDictionary, device_byteCompressedData, gpuMemoryOverflowIndex[i * 2], constMemoryFlag, gpuMemoryOverflowIndex[i * 2 + 1]);
				cudaError_t error_kernel = cudaGetLastError();
				if (error!= cudaSuccess)
				    printf("Error 20 :: %s\n", cudaGetErrorString(error));


				if(gpuBitPaddingFlag[i] == 0){
					error = cudaMemcpy(& inputFileData[pos], device_inputFileData, (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
					if (error!= cudaSuccess)
				        printf("Error 21 :: %s\n", cudaGetErrorString(error));

					pos += (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8);
				}
				else{
					unsigned char temp_compByte = inputFileData[pos - 1];
					error = cudaMemcpy(& inputFileData[pos - 1], device_inputFileData, ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			        if (error!= cudaSuccess)
				        printf("Error 22 :: %s\n", cudaGetErrorString(error));

					inputFileData[pos - 1] = temp_compByte | inputFileData[pos - 1];
					pos +=  (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
				}
			}


			// free allocated memory
			cudaFree(device_inputFileData);
			cudaFree(device_compressedDataOffset);
			cudaFree(device_huffmanDictionary);
			cudaFree(device_byteCompressedData);
		}

		else{
			// additional variable to store offset data after integer oveflow
			unsigned char *device_byteCompressedDataOverflow;
			error = cudaMalloc((void **) & device_byteCompressedDataOverflow, (compressedDataOffset[integerOverflowIndex[0]]) * sizeof(unsigned char));
			if (error!= cudaSuccess)
                   printf("Error 23 :: %s\n", cudaGetErrorString(error));

			error = cudaMalloc((void **)& device_byteCompressedDataOverflow, compressedDataOffset[gpuMemoryOverflowIndex[1]] * sizeof(unsigned char));
			if (error!= cudaSuccess)
                printf("Error 22 :: %s\n", cudaGetErrorString(error));

			// debug
			if(1){
				cudaMemGetInfo(& memoryFree, & memoryTotal);
				printf("Free Mem: %zu\n", memoryFree);
			}

			unsigned int pos = 0;
			for(unsigned int i = 0; i < numberOfKernels; i++){
				if(integerOverflowIndex[i] != 0){

					// initialize device_byteCompressedData
					error = cudaMemset(device_byteCompressedData, 0, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
					if (error!= cudaSuccess)
				        printf("Error 22 :: %s\n", cudaGetErrorString(error));

					error = cudaMemset(device_byteCompressedDataOverflow, 0, compressedDataOffset[gpuMemoryOverflowIndex[1]] * sizeof(unsigned char));
					if (error!= cudaSuccess)
				        printf("Error 23 :: %s\n", cudaGetErrorString(error));

					compress<<<1, BLOCK_SIZE>>>(device_inputFileData, device_compressedDataOffset, device_huffmanDictionary, device_byteCompressedData, device_byteCompressedDataOverflow, gpuMemoryOverflowIndex[i * 2], constMemoryFlag, gpuMemoryOverflowIndex[i * 2 + 1], integerOverflowIndex[i]);
					cudaError_t error_kernel = cudaGetLastError();
					if (error_kernel != cudaSuccess)
						printf("Error kernel 3 :: %s\n", cudaGetErrorString(error_kernel));

					if(gpuBitPaddingFlag[i] == 0){
						if(bitPaddingFlag[i] == 0){
							error = cudaMemcpy(& inputFileData[pos], device_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 24 :: %s\n", cudaGetErrorString(error));

							error = cudaMemcpy(& inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8)], & device_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 25 :: %s\n", cudaGetErrorString(error));

							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8);
						}
						else{
							error = cudaMemcpy(& inputFileData[pos], device_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 26 :: %s\n", cudaGetErrorString(error));

							unsigned char temp_compByte = inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];

							error = cudaMemcpy(& inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1], & device_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 27 :: %s\n", cudaGetErrorString(error));

							inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1] = temp_compByte | inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];
							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
						}
					}

					// padding is done
					else{
						unsigned char temp_gpuCompByte = inputFileData[pos - 1];

						if(bitPaddingFlag[i] == 0){
							error = cudaMemcpy(&inputFileData[pos - 1], device_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 28 :: %s\n", cudaGetErrorString(error));

							error = cudaMemcpy(& inputFileData[pos -1 + (compressedDataOffset[integerOverflowIndex[i]] / 8)], & device_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 29 :: %s\n", cudaGetErrorString(error));

							inputFileData[pos - 1] = temp_gpuCompByte | inputFileData[pos - 1];
							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
						}
						else{
							error = cudaMemcpy(& inputFileData[pos - 1], device_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 30 :: %s\n", cudaGetErrorString(error));

							unsigned char temp_compByte = inputFileData[ pos -1 + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];

							error = cudaMemcpy(& inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8) - 1], & device_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
				                printf("Error 31 :: %s\n", cudaGetErrorString(error));

							inputFileData[(compressedDataOffset[pos - 1 + integerOverflowIndex[i]] / 8) - 1] = temp_compByte | inputFileData[pos - 1 + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];
							inputFileData[pos - 1] = temp_gpuCompByte | inputFileData[pos - 1];
							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 2;
						}
					}
				}
				else{
					// initialize device_byteCompressedData
					error = cudaMemset(device_byteCompressedData, 0, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
					if (error!= cudaSuccess)
                        printf("Error 32 :: %s\n", cudaGetErrorString(error));

					compress<<<1, BLOCK_SIZE>>>(device_inputFileData, device_compressedDataOffset, device_huffmanDictionary, device_byteCompressedData, gpuMemoryOverflowIndex[i * 2], constMemoryFlag, gpuMemoryOverflowIndex[i * 2 + 1]);
					cudaError_t error_kernel = cudaGetLastError();
					if (error_kernel != cudaSuccess)
						printf("Error Kernel 4 :: %s\n", cudaGetErrorString(error_kernel));


					if(gpuBitPaddingFlag[i] == 0){
						error = cudaMemcpy(& inputFileData[pos], device_inputFileData, (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
						if (error != cudaSuccess)
                            printf("Error 33 :: %s\n", cudaGetErrorString(error));

						pos += (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8);
					}
					else{
						unsigned char temp_huffmanTreeNode = inputFileData[pos - 1];
						error = cudaMemcpy(& inputFileData[pos - 1], device_inputFileData, ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
						if (error != cudaSuccess)
                            printf("Error 34 :: %s\n", cudaGetErrorString(error));

						inputFileData[pos - 1] = temp_huffmanTreeNode | inputFileData[pos - 1];
						pos +=  (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
					}
				}
			}

			// free allocated memory
			cudaFree(device_inputFileData);
			cudaFree(device_compressedDataOffset);
			cudaFree(device_huffmanDictionary);
			cudaFree(device_byteCompressedData);
		}
	}
}