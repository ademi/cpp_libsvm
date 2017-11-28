#ifndef SVM_CONTEXT_H
#define SVM_CONTEXT_H

#include <vector>
#include "core/svm.h"

class svm_context {

	struct LIB_SVM::svm_model *_model;
	struct LIB_SVM::svm_problem *_problem;

	int _samplesNo;
	int _attributesNo;
	bool _initiated = false;
	
	double _scaling_mean = 1;
	double _scaling_std = 1;

public:
	struct LIB_SVM::svm_parameter *_params;
	svm_context();
	svm_context(int samplesNo, int featuresNo);
	svm_context(std::vector<std::vector<double>> &data, std::vector<int> &labels);
	bool init(int samplesNo, int featuresNo);
	bool init(std::vector<std::vector<double>> &data, std::vector<int> &labels);
	bool update_data(std::vector<std::vector<double>> &data, std::vector<int> &labels);
	void release();
	virtual ~svm_context();

	void set_default_params();
	void scale_attributes(std::vector<double> &data);
	bool map_data_to_problem(std::vector<std::vector<double>> &data, std::vector<int> &labels);


	bool write_model_to_file(std::string file_name);
	bool read_model_from_file(std::string file_name);

	LIB_SVM::svm_model* generate_model();
	void make_sample(std::vector<double> attr,std::vector<LIB_SVM::svm_node> &nodes);
	void predict(std::vector<double>const &attr, double *prediction, double *probability);
	//double getPrecsion();

};

#endif // !SVM_CONTEXT_H