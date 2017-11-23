#include <iostream>

#include "svm_context.h"


int main(int argsc, char** args) {
	// XOR test
	
	svm_context _svm;

	std::vector<std::vector<double>> attribute(4, std::vector<double>(2));
	attribute[0][0] = 0;
	attribute[0][1] = 0;
	attribute[1][0] = 0;
	attribute[1][0] = 1;
	attribute[2][0] = 1;
	attribute[2][0] = 0;
	attribute[3][0] = 1;
	attribute[3][0] = 1;

	std::vector<int> label(4);
	label[0] = 0;
	label[1] = 1;
	label[2] = 1;
	label[3] = 0;

	_svm.init(attribute, label);

	_svm.generate_model();

	std::cout << "#### svm model generated"<<std::endl;
	std::vector<double>sample(2);
	std::vector<LIB_SVM::svm_node>nodes(2);
	double prediction, prob;
	_svm.make_sample(sample, nodes);
	_svm.predict(nodes.data(), &prediction, &prob);

	std::cout << "prediction: " << prediction << "\n probability: " << prob << std::endl;
	int i = 0;
	std::cin >> i;
	return 0;
}