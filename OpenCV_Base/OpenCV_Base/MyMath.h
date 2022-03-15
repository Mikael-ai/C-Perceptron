#pragma once
#include <random>
#include <vector>
#include <iostream>
#include <ctime>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace mm{
	using std::vector;
	
	//Read-Write
	void writeVectorToFile(vector<double>& vec, std::string& path);
	void writeMatrixToFile(vector<vector<double>>& matrix, std::string& path);
	void writeDataToFile(vector<vector<double>>& w1, vector<vector<double>>& w2, vector<double>& b1, vector<double>& b2);

	void readMatrixFromFile(vector<vector<double>>& matrix, std::string& path);
	void readVectorFromFile(vector<double>& vec, std::string& path);
	void readDataFromFile(vector<vector<double>>& w1, vector<vector<double>>& w2, vector<double>& b1, vector<double>& b2);

	//Parameters
	const int _ImageSize = 36;
	const int _InputSize = _ImageSize * _ImageSize;

	//Generates random number
	double fRand(double fMin, double fMax);
	void initializeVectorWithRandom(vector<double>& vector);
	void initializeMatrixWithRandom(vector<vector<double>>& vector);

	//Image operations (openCV)
	void resizeImage(cv::Mat& image, const int& size);
	void convertRGBtoGrayscale(cv::Mat& image);
	vector<double> initializeImageVector(cv::Mat& image);
	
	//Matrix operations
	void multiplyMatrix(vector<vector<double>>& aMatrix, vector<vector<double>>& bMatrix, vector<vector<double>>& product);
	void multiplyMatrix(vector<double>& aMatrix, vector<vector<double>>& bMatrix, vector<double>& product);
	void sumVectors(vector<double>& aVector, vector<double>& bVector);
	vector<vector<double>> transposeMatrix(vector<vector<double>>& matrix);

	//Non-liniar function
	vector<double> Sigm(vector<double>& vector);
	void normalizeInput(vector<double>& input);

	//Neuron operations 
	void correctOutputWeights(vector<vector<double>>& Wn, vector<double>& Hn, vector<double>& error, double& learningRate);
	void correctOutputBias(vector<double>& bias, vector<double>& Hn, vector<double>& error, double& learningRate);
	vector<double> getHiddenError(vector<vector<double>>& Wn, vector<double>& Hn, vector<double>& outputError);
	vector<double> getOutputError(vector<double> &outputLayer, vector<double> &expected);
	void correctHiddenWeights(vector<double>& input, vector<vector<double>>& Wn, vector<double>& Hn, vector<double>& hiddenError);
	void correctHiddenBias(vector<double>& input, vector<double>& bias, vector<double>& Hn, vector<double>& hiddenError);

	//Display
	template<typename T>
	void dislplayVector(vector<T>& vector);

	void overallCorrection(cv::Mat image, vector<double>& x, vector<vector<double>>& w1, vector<vector<double>>& w2,
		vector<double>& b1, vector<double>& b2, vector<double>& h, vector<double>& o1, vector<double>& expected,
		vector<double>& outputError, vector<double>& hiddenError);
}

namespace nn {
	using std::vector;
	using std::cout;
	using std::endl;
	
	//Layers parameters 
	const uint32_t input_DIM = mm::_InputSize;
	const uint32_t hidden_DIM = 36;
	const uint32_t out_DIM = 2;

	double learningRate = 0.999;

	//weights [N x M]
	vector<vector<double>> w1(input_DIM, vector<double>(hidden_DIM));
	vector<vector<double>> w2(hidden_DIM, vector<double>(out_DIM));

	//bias vectors
	vector<double> b1(hidden_DIM);
	vector<double> b2(out_DIM);

	//Errors
	vector<double> outputError(out_DIM);
	vector<double> hiddenError(hidden_DIM);

	//Teacher
	vector<double> expected(out_DIM);

	//h = simg(x * W + b1)
	vector<double> h(hidden_DIM);
	//o = simg(h * W + b2)
	vector<double> o1(out_DIM);

	// Batch size
	uint32_t teaching_parameters_size = 100;
	vector <vector<double>> expectedVector(teaching_parameters_size);
	
	void study();
	vector<double> predict(std::string path);
}