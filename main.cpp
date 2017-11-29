#include <iostream>

#include "svm_context.h"

void xor_test();
int main(int argsc, char** args) {

	xor_test();
	getchar();
	return 0;
}
void xor_test() {
	// XOR test

	// create the matrix of xor truth table
	// inputs:
	std::vector<std::vector<double>> attribute(4, std::vector<double>(2));
	attribute[0][0] = 0.0;
	attribute[0][1] = 0.0;
	attribute[1][0] = 0.0;
	attribute[1][1] = 1.0;
	attribute[2][0] = 1.0;
	attribute[2][1] = 0.0;
	attribute[3][0] = 1.0;
	attribute[3][1] = 1.0;

	// outputs:
	std::vector<int> label(4);
	label[0] = 0;
	label[1] = 1;
	label[2] = 1;
	label[3] = 0;

	// initialize svm object with default settings (edit the svm_context::param to change any parameters)
	// ideally those settings (especially the kernel parameters) should come from svm training using libsvm library

	svm_context _svm;
	if (_svm.init(attribute, label)) {

		// create svm model using default settings
		LIB_SVM::svm_model *model = _svm.generate_model();

		//testing the resulting model with testing node
		std::vector<double>sample(2);
		sample[0] = 1;
		sample[1] = 0;

		double prediction = 0;
		std::vector<double> probability(2);

		_svm.predict_probability(sample, prediction, probability);
		_svm.predict(sample, prediction);

		//display the results to standard output
		// libsvm creates probability value for each label and assigns the prediction
		// to the label with maximum probability

		std::cout
			<< "\n***********************************\n"
			<< "prediction: " << prediction
			<< "\nprobability: " << probability[0] << "\t" << probability[1] << std::endl;

	}
	else
		std::cout << "error in initializing svm" << std::endl;
}