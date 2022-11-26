#include<GAME/GAME.h>
#include<vector>
#include<iostream>

#define lenofarray 5

using namespace std;
using namespace GAME;

void printVector(vector<float> data) {
	for(int i = 0; i < data.size(); i++) {
		cout<<data[i]<<"\t";
	}
	cout<<endl;
}

vector<float> generateRandomNumberArray(int n) {
	vector<float> a;
	for(int i = 0; i < n; i++) {
		a.push_back((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
	}
	return a;
}

int main(int argc, char** argv) {
	engine* cuengine = new engine(GAME_CUDA);
	cout<<"Initialized CUDA Engine..."<<endl;
	engine* clengine = new engine(GAME_OPENCL);
	cout<<"Initialized OpenCL Engine..."<<endl;

	vector<float> inputVector1 = generateRandomNumberArray(lenofarray);
	vector<float> inputVector2 = generateRandomNumberArray(lenofarray);

	cout<<"\n\nInput Vectors:"<<endl;
	printVector(inputVector1);
	printVector(inputVector2);

	memobj cudata1 = cuengine->createDataObject(inputVector1);
	memobj cudata2 = cuengine->createDataObject(inputVector2);
	memobj cudataa = cuengine->addVectors(cudata1, cudata2);
	memobj cudatas = cuengine->subtractVectors(cudata1, cudata2);
	memobj cudatam = cuengine->multiplyVectors(cudata1, cudata2);
	memobj cudatad = cuengine->divideVectors(cudata1, cudata2);

	cout<<"\n\nOutput computed on CUDA"<<endl;
	cout<<"Addition      :\t  ";printVector(cuengine->getDataObject(cudataa));
	cout<<"Subtraction   :\t  ";printVector(cuengine->getDataObject(cudatas));
	cout<<"Multiplication:\t  ";printVector(cuengine->getDataObject(cudatam));
	cout<<"Division      :\t  ";printVector(cuengine->getDataObject(cudatad));

	memobj cldata1 = clengine->createDataObject(inputVector1);
	memobj cldata2 = clengine->createDataObject(inputVector2);
	memobj cldataa = clengine->addVectors(cldata1, cldata2);
	memobj cldatas = clengine->subtractVectors(cldata1, cldata2);
	memobj cldatam = clengine->multiplyVectors(cldata1, cldata2);
	memobj cldatad = clengine->divideVectors(cldata1, cldata2);

	cout<<"\n\nOutput computed on OpenCL"<<endl;
	cout<<"Addition      :\t  ";printVector(clengine->getDataObject(cldataa));
	cout<<"Subtraction   :\t  ";printVector(clengine->getDataObject(cldatas));
	cout<<"Multiplication:\t  ";printVector(clengine->getDataObject(cldatam));
	cout<<"Division      :\t  ";printVector(clengine->getDataObject(cldatad));
}