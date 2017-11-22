#ifndef SVM_CONTEXT_CPP
#define SVM_CONTEXT_CPP

#include "svm_context.h"

svm_context::svm_context() {
	_model = nullptr;
	_problem = nullptr;
	_param = nullptr;

	_initiated = false;
}

svm_context::svm_context(int samples_nr, int attributes_nr) {
	_initiated = init(samples_nr, attributes_nr);
}

svm_context::svm_context(std::vector<std::vector<double>> data, std::vector<int> labels) {
	init(data, labels);
}
svm_context::~svm_context() {
	release();
}
bool svm_context::init(int samplesNo, int attributesNo) {
	if (!_initiated) {
		_samplesNo = samplesNo;
		_attributesNo = attributesNo;


		// Memory Allocation
		_problem->y = new double[_samplesNo];
		_problem->x = new LIB_SVM::svm_node*[_samplesNo];
		for (int i = 0; i < _samplesNo; ++i)
			_problem->x[i] = new LIB_SVM::svm_node[_attributesNo + 1];


		_initiated = true;

		return true;
	}
	return false;
}
/*
*	initiates lib svm problem using a vector of vector that contains the training samples and a vector of integers contains
*	the labels of the training samples.
*
*	LIBSVM omit an index and value pairs of a in case the value of the attribute is 0. However, the caller to this function needs to restore
*	the attribute pair.
*	expected format:
*	data:
* labels   attr[0]	att[1]	attr[2]	attr[3]	attr[4]
*	1       0        0.1      0.2      0        0
*	2		0        0.1      0.3     -1.2      0
*	1       0.4      0        0        0        0
*	2       0        0.1      0        1.4      0.5
*	3      -0.1     -0.2      0.1      1.1      0.1

*	data -> {{0,0.1,0.2,0,0} , {0,0.1,0.3,-1.2,0} , {0.4,0,0,0,0} , {0,0.1,0.1.4,0.5},{-0.1,0.2,0.1,1.1,0.1}}
*	labels-> {1,2,1,2,3}
*/
bool svm_context::init(std::vector<std::vector<double>> data, std::vector<int> labels) {
	
	// if there is no data return;
	if (data.size() < 1 || labels.size() < 1)return false;
	if (!_initiated) {
		if (!init(data.size(), data.at(0).size()))
			return;
	}

	for (auto data_it = data.begin(); data_it != data.end(); ++data_it) {
		int row = data_it - data.begin();
		
		for (auto attribute_it = data_it->begin(); attribute_it != data_it->end(); ++attribute_it) {
			
			int col = attribute_it - data_it->begin();
			_problem->x[row][col].index = col + 1;
			_problem->x[row][col].value = *(attribute_it);
		}

		int last_index = data_it->end() - data_it->begin() + 1;
		_problem->x[row][last_index].index = -1;
		_problem->x[row][last_index].value = 0;
	}
}
void svm_context::release() {
	if (!_initiated)return;

	LIB_SVM::svm_destroy_param(_param);
	for (int row = 0; row < _problem->l; row++) {
		delete[]_problem->x[row];
		_problem->x[row] = nullptr;
	}

	delete[]_problem->x;
	_problem->x = nullptr;
	delete[]_problem->y;
	_problem->y = nullptr;
	if (_model)svm_free_and_destroy_model(&_model);
}
#endif // !SVM_CONTEXT_CPP



