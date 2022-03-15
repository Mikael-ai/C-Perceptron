#include "MyMath.h"
#include <fstream>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

namespace mm {
	void writeVectorToFile(vector<double>& vec, std::string& path) {
		std::ofstream fout(path);
		if (!fout.is_open()) {
			std::cout << "Can't open the file...";
		}

		for (int i = 0; i < vec.size(); i++) {
			fout << vec[i];
			if (i != vec.size() - 1) {
				fout << " ";
			}
		}
		fout.close();
	}

	void writeMatrixToFile(vector<vector<double>>& matrix, std::string& path) {
		std::ofstream fout(path);
		if (!fout.is_open()) {
			std::cout << "Can't open the file...";
		}

		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				fout << matrix[i][j];
				if (j != matrix[0].size() - 1) {
					fout << " ";
				}
			}
			if (i != matrix.size()) {
				fout << '\n';
			}
		}
		fout.close();
	}

	void writeDataToFile(vector<vector<double>>& w1, vector<vector<double>>& w2, vector<double>& b1, vector<double>& b2) {
		std::string path = fs::current_path().string();

		std::string w1_path = path + "\\Weights\\w1.txt";
		std::string w2_path = path + "\\Weights\\w2.txt";

		std::string b1_path = path + "\\Weights\\b1.txt";
		std::string b2_path = path + "\\Weights\\b2.txt";

		writeMatrixToFile(w1, w1_path);
		writeMatrixToFile(w2, w2_path);

		writeVectorToFile(b1, b1_path);
		writeVectorToFile(b2, b2_path);
	}

	void readMatrixFromFile(vector<vector<double>>& matrix, std::string& path) {
		std::ifstream fin(path);
		if (!fin.is_open()) {
			std::cout << "Can't open the file...";
		}
		std::string str;

		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				if (!fin.eof()) {
					str = "";
					fin >> str;
					matrix[i][j] = std::stod(str);
				}
			}
		}
		fin.close();
	}

	void readVectorFromFile(vector<double>& vec, std::string& path) {
		std::ifstream fin(path);
		if (!fin.is_open()) {
			std::cout << "Can't open the file...";
		}
		std::string str;

		for (int i = 0; i < vec.size(); i++) {
			if (!fin.eof()) {
				str = "";
				fin >> str;
				vec[i] = std::stod(str);
			}
		}
		fin.close();
	}

	void readDataFromFile(vector<vector<double>>& w1, vector<vector<double>>& w2, vector<double>& b1, vector<double>& b2) {
		std::string path = fs::current_path().string();

		std::string w1_path = path + "\\Weights\\w1.txt";
		std::string w2_path = path + "\\Weights\\w2.txt";

		std::string b1_path = path + "\\Weights\\b1.txt";
		std::string b2_path = path + "\\Weights\\b2.txt";

		readMatrixFromFile(w1, w1_path);
		readMatrixFromFile(w2, w2_path);

		readVectorFromFile(b1, b1_path);
		readVectorFromFile(b2, b2_path);
	}
	
	double fRand(double fMin, double fMax) {
		double f = (double)rand() / RAND_MAX;
		return fMin + f * (fMax - fMin);
	}

	void resizeImage(cv::Mat& image, const int& size) {
		cv::resize(image, image, cv::Size(size, size));
	};

	void convertRGBtoGrayscale(cv::Mat& image) {
		cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	}
	
	vector<double> initializeImageVector(cv::Mat& image) {
		vector<double> imageVector;
		if (image.isContinuous()) {
			imageVector.assign(image.datastart, image.dataend);
		}
		else {
			for (int i = 0; i < image.rows; ++i) {
				imageVector.insert(imageVector.end(), image.ptr<uchar>(i), image.ptr<uchar>(i) + image.cols);
			}
		}
		return imageVector;
	}

	template<typename T>
	void dislplayVector(vector<T>& vector) {
		for (T i : vector) {
			std::cout << i << " ";
		}
	}

	void normalizeInput(vector<double>& input) {
		double sum = 0;
		for (int i = 0; i < input.size(); i++) {
			sum += input[i] * input[i];
		}
		double squaredSum = sqrt(sum);
		for (int i = 0; i < input.size(); i++) {
			input[i] = input[i] / squaredSum;
		}
	}

	//Matrix * matrix
	void multiplyMatrix(vector<vector<double>>& aMatrix, vector<vector<double>>& bMatrix, vector<vector<double>>& product) {
		for (int row = 0; row < product.size(); row++) {
			for (int col = 0; col < product[0].size(); col++) {
				for (int inner = 0; inner < aMatrix[0].size(); inner++) {
					product[row][col] += aMatrix[row][inner] * bMatrix[inner][col];
				}
				//std::cout << product[row][col] << "  ";
			}
			//std::cout << "\n";
		}
	}

	//Vector * matrix
	void multiplyMatrix(vector<double>& aMatrix, vector<vector<double>>& bMatrix, vector<double>& product) {
		for (int col = 0; col < bMatrix[0].size(); col++) {
			for (int inner = 0; inner < aMatrix.size(); inner++) {
				product[col] += aMatrix[inner] * bMatrix[inner][col];
			}
			//std::cout << product[col] << "  ";
		}
		//std::cout << "\n";
	}

	//Transpose matrix
	vector<vector<double>> transposeMatrix(vector<vector<double>>& matrix) {
		vector<vector<double>> transposed(matrix[0].size(), vector<double>(matrix.size()));

		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				transposed[j][i] = matrix[i][j];
			}
		}

		return transposed;
	}

	void sumVectors(vector<double>& aVector, vector<double>& bVector) {
		for (int i = 0; i < aVector.size(); i++) {
			aVector[i] += bVector[i];
		}
	}

	vector<double> Sigm(vector<double>& vector) {
		for (int i = 0; i < vector.size(); i++) {
			vector[i] = 1 / (1 + exp(-vector[i]));
		}
		return vector;
	}

	void initializeVectorWithRandom(vector<double>& vector) {
		for (int i = 0; i < vector.size(); i++) {
			vector[i] = fRand(-1, 1);
		}
	}

	void initializeMatrixWithRandom(vector<vector<double>>& vector) {
		for (int i = 0; i < vector.size(); i++) {
			for (int j = 0; j < vector[0].size(); j++) {
				vector[i][j] = fRand(-1, 1);
			}
		}
	}

	void correctOutputWeights(vector<vector<double>>& Wn, vector<double>& Hn, vector<double> &error, double& learningRate) {
		for (int i = 0; i < Wn.size(); i++) {
			for (int j = 0; j < Wn[0].size(); j++) {
				Wn[i][j] = (Wn[i][j] + error[j] * Hn[i]) * learningRate;
			}
		};
	}

	void correctOutputBias(vector<double> &bias, vector<double>& Hn, vector<double>& error, double& learningRate) {
		for (int i = 0; i < bias.size(); i++) {
			bias[i] = (bias[i] + error[i] * Hn[i]) * learningRate;
		}
	}

	vector<double> getOutputError(vector<double> &outputLayer, vector<double> &expected) {
		vector<double> error(outputLayer.size());
		for (int i = 0; i < outputLayer.size(); i++) {
			error[i] = (expected[i] - outputLayer[i]) * outputLayer[i] * (1 - outputLayer[i]);
		}

		return error;
	}

	vector<double> getHiddenError(vector<vector<double>>& Wn, vector<double>& Hn, vector<double>& outputError) {

		vector<double> betaHiddenError(Wn.size());
		vector<double> hiddenError(Wn.size());
		vector<vector<double>> transposed_Wn = transposeMatrix(Wn);
		multiplyMatrix(outputError, transposed_Wn, betaHiddenError);

		for (int i = 0; i < betaHiddenError.size(); i++) {
			hiddenError[i] = Hn[i] * (1 - Hn[i]) * betaHiddenError[i];
		}

		return hiddenError;
	}

	void correctHiddenWeights(vector<double> &input, vector<vector<double>>& Wn, vector<double>& Hn, vector<double>& hiddenError) {
		for (int i = 0; i < Wn.size(); i++) {
			for (int j = 0; j < Wn[0].size(); j++) {
				Wn[i][j] = Wn[i][j] + hiddenError[j] * input[i];
			}
		}
	}

	void correctHiddenBias(vector<double>& input, vector<double>& bias, vector<double>& Hn, vector<double>& hiddenError) {
		for (int i = 0; i < bias.size(); i++) {
			for (int j = 0; j < input.size(); j++) {
				bias[i]  += hiddenError[i] * input[j];
			}	
		}
	}

	void overallCorrection(cv::Mat image, vector<double>& x, vector<vector<double>>& w1, vector<vector<double>>& w2,
		vector<double>& b1, vector<double>& b2, vector<double>& h, vector<double>& o1, vector<double>& expected,
		vector<double>& outputError, vector<double>& hiddenError) {

		double learningRate = nn::learningRate;

		outputError = mm::getOutputError(o1, expected);

		//Hidden error 
		// Hn * (1 - Hn) * (output error * transposed(Wn))
		hiddenError = mm::getHiddenError(w2, h, outputError);

		//Correct output weighs and bias
		mm::correctOutputWeights(w2, h, outputError, learningRate);
		mm::correctOutputBias(b2, h, outputError, learningRate);

		//Correct hidden weighs and bias
		mm::correctHiddenWeights(x, w1, h, hiddenError);
		mm::correctHiddenBias(x, b1, h, hiddenError);
	}
}

namespace nn {
	void study() {
		//Relation between our random function and time
		srand((unsigned)time(NULL));
		srand((unsigned)rand());

		//Teacher
		int num;
		for (int i = 0; i < teaching_parameters_size; i++) {
			num = 0;
			if (i % 2 == 0) {
				expectedVector[i].push_back(num + 1);
				expectedVector[i].push_back(num);
			}
			else {
				expectedVector[i].push_back(num);
				expectedVector[i].push_back(num + 1);
			}
		} 

		//Declare image
		cv::Mat image;

		//Creating an image vector (Our input)
		vector<double> x(mm::_InputSize); 
		
		//Random weights 
		mm::initializeMatrixWithRandom(w1);
		mm::initializeMatrixWithRandom(w2);

		//Random bias
		mm::initializeVectorWithRandom(b1);
		mm::initializeVectorWithRandom(b2);

		//Get count of files 
		std::string path = "C:\\Users\\mikik\\Desktop\\Pes\\Combo";

		vector<std::string> fileDirectory;
		for (auto& p : fs::directory_iterator(path)) {
			fileDirectory.push_back(p.path().string());
		}
		
		int fileCounter = -1;
		uint32_t epochCounter = 0;
		bool isDone = false;

		double errorSum_0 = 0;
		double errorSum_1 = 0;
		double sumE0 = 1;
		double sumE1 = 1;

		while (!isDone) {
			fileCounter = fileCounter + 1;

			if (fileCounter >= teaching_parameters_size) {
				fileCounter = 0;
				errorSum_0 = 0;
				errorSum_1 = 0;
				epochCounter += 1;
			}

			//Read image file
			image = cv::imread(fileDirectory[fileCounter]);
			if (image.empty()) {
				cout << "ERRROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR" << endl;
				std::cin.get();
			}

			//Convert to grayscale
			mm::convertRGBtoGrayscale(image);

			//input vector
			x = mm::initializeImageVector(image);
			mm::normalizeInput(x);

			//h = simg(x * Wn + bias)
			mm::multiplyMatrix(x, w1, h);
			mm::sumVectors(h, b1);
			mm::Sigm(h);
			
			//o = simg(h * W + b2)
			mm::multiplyMatrix(h, w2, o1);
			mm::sumVectors(o1, b2);
			mm::Sigm(o1);

			//Correction
			if (fileCounter % 2 == 0) {
				if (o1[0] < 0.8 || o1[1] > o1[0]) {
					mm::overallCorrection(image, x, w1, w2, b1, b2, h, o1, expectedVector[fileCounter], outputError, hiddenError);
				}
			}
			if (fileCounter % 2 != 0) {
				if (o1[1] < 0.8 || o1[0] > o1[1]) {
					mm::overallCorrection(image, x, w1, w2, b1, b2, h, o1, expectedVector[fileCounter], outputError, hiddenError);
				}
			}
			
			errorSum_0 += abs(expectedVector[fileCounter][0] - o1[0]);
			errorSum_1 += abs(expectedVector[fileCounter][1] - o1[1]);

			if (fileCounter == teaching_parameters_size - 1) {
				sumE0 = errorSum_0 / teaching_parameters_size;
				sumE1 = errorSum_1 / teaching_parameters_size;

				cout << "Error 0: " << sumE0 << " | Error 1: " << sumE1 << endl;
			}

			//Display info
			cout << "Image number " << fileCounter << "\n";
			cout << "Epoch: " << epochCounter << endl;
				for (int i = 0; i < o1.size(); i++) {
					cout << "O[" << i << "] = " << o1[i] << "\n";
				}

			cout << "Expected O[0]: " << expectedVector[fileCounter][0]
				<< " | Expected O[1]: " << expectedVector[fileCounter][1] << "\n" << endl;

			//if (isDone) cout << "DONE!!!!!!!!!!!!!!!!!!!!!!!!!\n";
			if (fileCounter == teaching_parameters_size - 1 && (sumE0 <= 0.1 && sumE1 <= 0.1)) {
				isDone = true;
				cout << "ALMOST DONE !!! Writing data, wait a litte...\n";
				mm::writeDataToFile(w1, w2, b1, b2);
				cout << "DONE !!!\n";
			}
		}
	}

	vector<double> predict(std::string imagePath) {
		//Get weights and bias from set
		// UNCOMMENT THIS ONE RIGHT BELOW
		mm::readDataFromFile(w1, w2, b1, b2);
		
		//Read image file
		cv::Mat image = cv::imread(imagePath);
		if (image.empty()) {
			cout << "ERRROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR" << endl;
			std::cin.get();
		}

		//Resize and convert to grayscale
		mm::resizeImage(image, mm::_ImageSize);
		mm::convertRGBtoGrayscale(image);

		//input vector
		vector<double>x(mm::_InputSize);
		x = mm::initializeImageVector(image);
		mm::normalizeInput(x);

		//h = simg(x * Wn + bias)
		mm::multiplyMatrix(x, w1, h);
		mm::sumVectors(h, b1);
		mm::Sigm(h);

		//o = simg(h * W + b2)
		mm::multiplyMatrix(h, w2, o1);
		mm::sumVectors(o1, b2);
		mm::Sigm(o1);
		
		if (o1[0] > o1[1]) {
			cout << "RESULT: It's a CAT" << ": " << int(o1[0]*100) << "%" << endl;
			//cout << "Cat: " << o1[0] << endl;
			//cout << "Dog: " << o1[1] << endl;
		}
		if (o1[1] > o1[0]) {
			//cout << "Cat: " << o1[0] << endl;
			//cout << "Dog: " << o1[1] << endl;
			cout << "RESULT: It's a DOG" << ": " << int(o1[1] * 100) << "%" << endl;
		}
		
		return o1;
	}
}

int main() {
	//COMMENT THIS IF IT'S ALREADY WAS TRAINED
	//nn::study();

	std::cout << "First one (Cat image):" << std::endl;
	nn::predict("D:\\JustAsPlanned\\cats\\2948_dhgPJtABzuA.jpg");

	std::cout << std::endl;

	std::cout << "Second one (DOG image):" << std::endl;
	nn::predict("C:\\Users\\mikik\\Desktop\\Pes\\Dog_train.jpg");

	std::cin.get();
	return 0;
}