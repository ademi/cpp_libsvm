#ifndef SVM_CONTEXT_CPP
#define SVM_CONTEXT_CPP

#include <numeric>
#include <algorithm>
#include <functional>
#include "svm_context.h"

svm_context::svm_context() {
	_model = nullptr;
	_problem = nullptr;

	_initiated = false;
}

svm_context::svm_context(int samples_nr, int attributes_nr) {
	_initiated = init(samples_nr, attributes_nr);
}

svm_context::svm_context(std::vector<std::vector<double>> &data, std::vector<int> &labels) {
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
		_problem = new LIB_SVM::svm_problem;
		_problem->y = new double[_samplesNo];
		_problem->x = new LIB_SVM::svm_node*[_samplesNo];
		for (int i = 0; i < _samplesNo; ++i)
			_problem->x[i] = new LIB_SVM::svm_node[_attributesNo + 1];

		set_default_params();
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
bool svm_context::update_data(std::vector<std::vector<double>> &data, std::vector<int> &labels) {
	if(!_initiated)return false;
	if (!_problem)return false;

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
	for (auto label_it = labels.begin(); label_it != labels.end(); ++label_it) {
		_problem->y[label_it - labels.begin()] = *label_it;
	}
	return true;
}
bool svm_context::init(std::vector<std::vector<double>> &data, std::vector<int> &labels) {
	
	// if there is no data return;
	if (data.size() < 1 || labels.size() < 1)return false;
	if (!_initiated) {
		if (!init(data.size(), data.at(0).size()))
			return false;
	}
	return update_data(data, labels);
}
void svm_context::set_default_params() {
	_params = new LIB_SVM::svm_parameter;
	//set all default parameters for param struct
	_params->svm_type = LIB_SVM::C_SVC;
	_params->kernel_type = LIB_SVM::RBF;
	_params->degree = 3;
	_params->gamma = 0.5;
	_params->coef0 = 0;
	_params->nu = 0.5;
	_params->cache_size = 100;
	_params->C = 1;
	_params->eps = 1e-3;
	_params->p = 0.1;
	_params->shrinking = 1;
	_params->probability = 1;
	_params->nr_weight = 0;
	_params->weight_label = NULL;
	_params->weight = NULL;
}
void svm_context::release() {
	if (!_initiated)return;

	if (_params) LIB_SVM::svm_destroy_param(_params);
	if (_model)svm_free_and_destroy_model(&_model);

	for (int row = 0; row < _problem->l; row++) {
		delete[]_problem->x[row];
		_problem->x[row] = nullptr;
	}

	delete[]_problem->x;
	_problem->x = nullptr;
	delete[]_problem->y;
	_problem->y = nullptr;

	_initiated = false;
}

void svm_context::scale_attributes(std::vector<double> &data){
	double sum = std::accumulate(data.begin(), data.end(), 0.0);
	double mean = sum / data.size();

	std::vector<double> diff(data.size());
	std::transform(data.begin(), data.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double stdev = std::sqrt(sq_sum / data.size());

	std::transform(data.begin(), data.end(), data.begin(), std::bind2nd(std::minus<double>(), mean));
	std::transform(data.begin(), data.end(), diff.begin(), std::bind2nd(std::divides<double>(), stdev));
}

bool svm_context::map_data_to_problem(std::vector<std::vector<double>> &data, std::vector<int> &labels) {
	
	//scale of the attributes
	for (auto it = data.begin(); it != data.end(); ++it) {
		scale_attributes(*it);
	}
	// map the scaled data to the svm problem
	return init(data, labels);
}

bool svm_context::write_model_to_file(std::string file_name) {
	if (!_model) return false;
	if (LIB_SVM::svm_save_model(file_name.c_str(), _model) != 0)
		return false;
	else return true;
}
bool svm_context::read_model_from_file(std::string file_name) {
	if (_model)
		svm_free_and_destroy_model(&_model);

	_model = LIB_SVM::svm_load_model(file_name.c_str());

	if (!_model) return false;
}

LIB_SVM::svm_model* svm_context::generate_model() {
	_model = svm_train(_problem, _params);
	return _model;
}

void svm_context::make_sample(std::vector<double> attr, std::vector<LIB_SVM::svm_node>&nodes) {
	if (_attributesNo == 0 || attr.size() != _attributesNo) nodes.resize(_attributesNo);

	for (auto it = attr.begin(); it != attr.end(); ++it) {
		int i = it - attr.begin();
		nodes.at(i).index = i;
		nodes.at(i).value = *(it);
	}
}
void  svm_context::predict(LIB_SVM::svm_node *sample, double *prediction, double *probability) {
	*prediction = LIB_SVM::svm_predict_probability(_model, sample, probability);
}
#endif // !SVM_CONTEXT_CPP



