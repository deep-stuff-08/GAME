#include<GAME/GAME.h>
#include<vector>
#include<iostream>

using namespace std;
using namespace GAME;

void printVector(vector<float> data) {
	for(int i = 0; i < data.size(); i++) {
		cout<<data[i]<<"\t";
	}
	cout<<endl;
}

int main(int argc, char** argv) {
	engine* clengine = new engine(GAME_OPENCL);

	memobj data1 = clengine->createDataObject({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
	memobj data2 = clengine->createDataObject({5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
	memobj dataa = clengine->addVectors(data1, data2);
	memobj datas = clengine->subtractVectors(data1, data2);
	memobj datam = clengine->multiplyVectors(data1, data2);
	memobj datad = clengine->divideVectors(data1, data2);

	printVector(clengine->getDataObject(dataa));
	printVector(clengine->getDataObject(datas));
	printVector(clengine->getDataObject(datam));
	printVector(clengine->getDataObject(datad));
}