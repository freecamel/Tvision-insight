#include "OpenCVFaceRecognition.h"
#include "OpenCVFaceRecognition.h"
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;


OpenCVFaceRecognition::OpenCVFaceRecognition()
{
}


OpenCVFaceRecognition::~OpenCVFaceRecognition(void)
{
}


static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

void OpenCVFaceRecognition::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') 
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
			 cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
			 images.push_back(img) ;
       //     images.push_back(imread(path, 0));
		//	cv::waitKey(5000);
		//	images.push_back( Mat(cvLoadImage(path.c_str(),  0)) );
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void OpenCVFaceRecognition::TrainModel(const string &trainFile)
{
//	trainFileName = "E:/PythonScript/att.txt" ; // default file name 
	vector<Mat> images;
    vector<int> labels;

	// Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(trainFile, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << trainFile << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

	// load in image 
	  if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
 //   Mat testSample = images[images.size() - 1];
  //  int testLabel = labels[labels.size() - 1];
   // images.pop_back();
    //labels.pop_back();

//	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
 //   model->train(images, labels);

	 model = cv::createEigenFaceRecognizer();
     model->train(images, labels);

	 cout << "train is done!" <<endl;
}

int  OpenCVFaceRecognition::Prediction(Mat &image) 
{
	    int predictedLabel = model->predict(image);

		return predictedLabel ;
}
