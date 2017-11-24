#include <iostream>

#include "svm_context.h"


int main(int argsc, char** args) {
	// XOR test
	
	svm_context _svm;

	std::vector<std::vector<double>> attribute(4, std::vector<double>(2));
	attribute[0][0] = -1;
	attribute[0][1] = -1;
	attribute[1][0] = -1;
	attribute[1][1] = 1;
	attribute[2][0] = 1;
	attribute[2][1] = -1;
	attribute[3][0] = 1;
	attribute[3][1] = 1;

	std::vector<int> label(4);
	label[0] = -1;
	label[1] = 1;
	label[2] = 1;
	label[3] = -1;

	if (_svm.init(attribute, label)) {
		int k = 0;
		LIB_SVM::svm_model *model = _svm.generate_model();


		//std::cout << "#### svm model generated"<<std::endl;
		std::vector<double>sample(2);
		sample[0] = 1;
		sample[1] = -1;
		std::vector<LIB_SVM::svm_node>nodes(2);
		double prediction=0, prob=0;
		_svm.make_sample(sample, nodes);
		_svm.predict(nodes.data(), &prediction, &prob);

		std::cout << "prediction: " << prediction << "\n probability: " << prob << std::endl;
		int i = 0;
		std::cin >> i;
	}

	return 0;
}