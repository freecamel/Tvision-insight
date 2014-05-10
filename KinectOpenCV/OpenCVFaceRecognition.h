#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
using namespace cv ;
using namespace std ; 

class OpenCVFaceRecognition
{
public:
	OpenCVFaceRecognition(void);
	~OpenCVFaceRecognition(void);

	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator);
	void TrainModel(const string &trainFile);  // train a model 
	int Prediction(Mat &image) ;  // predict model, provide a result  

private:
	// string trainFileName ;

	Ptr<FaceRecognizer> model ;
};

