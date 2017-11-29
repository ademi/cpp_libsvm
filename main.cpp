#include <iostream>

#include "svm_context.h"


int main(int argsc, char** args) {
	// XOR test
	
	// Creating SVM model
	std::vector<std::vector<double>> attribute(4, std::vector<double>(2));
	attribute[0][0] = 0.0;
	attribute[0][1] = 0.0;
	attribute[1][0] = 0.0;
	attribute[1][1] = 1.0;
	attribute[2][0] = 1.0;
	attribute[2][1] = 0.0;
	attribute[3][0] = 1.0;
	attribute[3][1] = 1.0;

	std::vector<int> label(4);
	label[0] = 0;
	label[1] = 1;
	label[2] = 1;
	label[3] = 0;

	// initialize svm object with default settings
	for (int iter = 0; iter < 200000; iter++) {

		svm_context _svm;
		if (_svm.init(attribute, label)) {
				LIB_SVM::svm_model *model = _svm.generate_model();


			std::vector<double>sample(2);
			sample[0] = 1;
			sample[1] = 0;

			double prediction = 0;
			std::vector<double> probability(2);

			_svm.predict_probability(sample, prediction, probability);
			_svm.predict(sample, prediction);
			std::cout
				<< "\n***********************************\n"
				<< "prediction: " << prediction
				<< "\nprobability: " << probability[0] << "\t" << probability[1] << std::endl;

		}
		else
			std::cout << "error in initializing svm" << std::endl;
		_svm.release();
	}

	//getchar();
	return 0;
}