#pragma once
#include <opencv2/ml/ml.hpp> 

#include <iostream>  
#include <string>
#include "get_files.h"
#include "MyLBP.h"

using namespace std;
using namespace cv;

Mat CreatetrainingData(string path_0, vector<string>& files_name){

	vector<string> files_0;
	getFiles(path_0, files_0);
	cout << " num of files : " << files_0.size() << endl;

	Mat trainingDataMat(files_0.size(), 60, CV_32FC1, Scalar::all(0));
	for (int i = 0; i < files_0.size(); ++i)
	{
		Mat src = imread(files_0[i], 0);
		Mat dst;
		olbp(src, dst);
		Mat hist;
		My_calc_Hist(dst, hist);
		vector<float> v;
		v = Mat_<float>(hist);
		for (int j = 0; j < 60; ++j){
			trainingDataMat.at<float>(i, j) = v[j];
		}
	}
	files_name = files_0;
	return trainingDataMat;
}

int main()
{
	string path_training = "D:\\liyi\\Videos\\training_data";


	// 生成训练数据
	vector<string> files_name;
	Mat trainingDataMat_0 = CreatetrainingData(path_training, files_name);

	// Label{活鱼:0， 死鱼:1}
	Mat labelsMat(trainingDataMat_0.rows, 1, CV_32FC1, Scalar::all(0));
	for (size_t i = 0; i < trainingDataMat_0.rows; i++)
	{
		//cout << files_name[i][path_training.size()+1] << endl;
		if (files_name[i][path_training.size()+1] == '1'){ labelsMat.at<float>(i, 0) = 1; }
	}

	// 生成网络
	// Setup the BPNetwork  
	CvANN_MLP bp;

	// Set up BPNetwork's parameters
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10000, 0.01);//终止条件
	params.bp_dw_scale = 0.1;     // 权值更新率
	params.bp_moment_scale = 0.1; // 权值更新冲量

	// 定义网络大小: 3层{ 输入层 60 , 隐藏层30 , 输出层1 }
	Mat layerSizes = (Mat_<int>(1, 3) << 60,30,1);

	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM);

	// training
	cout << "training . . ." << endl;

	bp.train(trainingDataMat_0, labelsMat, Mat(), Mat(), params);

	cout << "train end" << endl;
	bp.save("bp_network.xml");  // 保存训练好的网络到.xml文件中

	// testing
	string path_testing_0 = "D:\\liyi\\Videos\\test_0";
	string path_testing_1 = "D:\\liyi\\Videos\\test_1";

	vector<string> files;
	Mat testingDataMat_1 = CreatetrainingData(path_testing_1, files);
	Mat testingDataMat_0 = CreatetrainingData(path_testing_0, files);


	Mat responseMat, responseMat1;

	// 测试
	bp.predict(testingDataMat_0, responseMat);
	bp.predict(testingDataMat_1, responseMat1);

	// 输出测试结果
	cout << "test 1:" << endl;
	for (size_t i = 0; i < responseMat.rows; i++)
	{
		cout << responseMat.at<float>(i, 0)<<endl;
	}
	cout << endl << "test 2:" << endl;
	for (size_t i = 0; i < responseMat1.rows; i++)
	{
		cout << responseMat1.at<float>(i, 0)<< endl;
	}
	cout << endl;
	return 0;
}

