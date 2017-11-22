#ifndef SVM_CONTEXT_H
#define SVM_CONTEXT_H

#include <vector>
#include "core/svm.h"

class svm_context {

	struct LIB_SVM::svm_model *_model;
	struct LIB_SVM::svm_problem *_problem;
	struct LIB_SVM::svm_parameter *_param;		

	int _samplesNo;
	int _attributesNo;
	bool _initiated = false;
public:
	svm_context();
	svm_context(int samplesNo, int featuresNo);
	svm_context(std::vector<std::vector<double>> data, std::vector<int> labels);
	bool init(int samplesNo, int featuresNo);
	bool init(std::vector<std::vector<double>> data, std::vector<int> labels);
	void release();


	virtual ~svm_context();

	void scaleFeatures();
	void saveModel();
	void loadModel();
	LIB_SVM::svm_model* train();
	void crossVlidation();
	void  predict();
	double getPrecsion();
	double static XORTest();
};

#endif // !SVM_CONTEXT_H