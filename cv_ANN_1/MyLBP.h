#pragma once
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

namespace cv{

	//---------------------------计算LBP----------------------
	// a 圆形LBP算子

	// src为输入图像，dst为输出图像，radius为半径，neighbor为计算当前点LBP所需的邻域像素点数，也就是样本点个数
	template <typename _Tp> static // 模板函数，根据不同的原始数据类型得到不同的结果
		inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors)
	{
		//get matrices
		Mat src = _src.getMat();
		// allocate memory for result因此不用在外部给_dst分配内存空间，输出数据类型都是int
		_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
		Mat dst = _dst.getMat();
		// zero
		dst.setTo(0);
		for (int n = 0; n < neighbors; n++)
		{
			// sample points 获取当前采样点
			float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n / static_cast<float>(neighbors));
			float y = static_cast<float>(radius)* cos(2.0*CV_PI*n / static_cast<float>(neighbors));
			// relative indices 下取整和上取整
			int fx = static_cast<int>(floor(x)); // 向下取整
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));  // 向上取整
			int cy = static_cast<int>(ceil(y));
			// fractional part 小数部分
			float tx = x - fx;
			float ty = y - fy;
			// set interpolation weights 设置四个点的插值权重
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 = tx  *      ty;
			// iterate through your data 循环处理图像数据
			for (int i = radius; i < src.rows - radius; i++)
			{
				for (int j = radius; j < src.cols - radius; j++)
				{
					// calculate interpolated value 计算插值，t表示四个点的权重和
					float t = w1*src.at<_Tp>(i + fy, j + fx) +

						w2*src.at<_Tp>(i + fy, j + cx) +

						w3*src.at<_Tp>(i + cy, j + fx) +

						w4*src.at<_Tp>(i + cy, j + cx);

					// floating point precision, so check some machine-dependent epsilon
					// std::numeric_limits<float>::epsilon()=1.192092896e-07F
					// 当t>=src(i,j)的时候取1，并进行相应的移位
					dst.at<int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) ||
						(std::abs(t - src.at<_Tp>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
				}
			}
		}
	}

	// 外部接口，根据不同的数据类型调用模板函数
	static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
	{
		int type = src.type();
		switch (type) {
		case CV_8SC1:   elbp_<char>(src, dst, radius, neighbors); break;
		case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
		case CV_16SC1:  elbp_<short>(src, dst, radius, neighbors); break;
		case CV_16UC1:  elbp_<unsigned short>(src, dst, radius, neighbors); break;
		case CV_32SC1:  elbp_<int>(src, dst, radius, neighbors); break;
		case CV_32FC1:  elbp_<float>(src, dst, radius, neighbors); break;
		case CV_64FC1:  elbp_<double>(src, dst, radius, neighbors); break;
		default:
			string error_msg = format("Using Circle Local Binary Patterns for feature extraction only works                                     on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
			CV_Error(CV_StsNotImplemented, error_msg);
			break;
		}
	}

	Mat elbp(InputArray src, int radius, int neighbors) {
		Mat dst;
		elbp(src, dst, radius, neighbors);
		return dst;
	}

	//b 原始LBP算子

	// 原始LBP算子只是计算8邻域内的局部二值模式
	template <typename _Tp> static

		void olbp_(InputArray _src, OutputArray _dst) {
		// get matrices
		Mat src = _src.getMat();
		// allocate memory for result
		_dst.create(src.rows - 2, src.cols - 2, CV_8UC1);
		Mat dst = _dst.getMat();
		// zero the result matrix
		dst.setTo(0);
		// calculate patterns
		for (int i = 1; i < src.rows - 1; i++)
		{
			for (int j = 1; j < src.cols - 1; j++)
			{
				_Tp center = src.at<_Tp>(i, j);
				unsigned char code = 0;
				code |= (src.at<_Tp>(i - 1, j - 1) >= center) << 7;
				code |= (src.at<_Tp>(i - 1, j) >= center) << 6;
				code |= (src.at<_Tp>(i - 1, j + 1) >= center) << 5;
				code |= (src.at<_Tp>(i, j + 1) >= center) << 4;
				code |= (src.at<_Tp>(i + 1, j + 1) >= center) << 3;
				code |= (src.at<_Tp>(i + 1, j) >= center) << 2;
				code |= (src.at<_Tp>(i + 1, j - 1) >= center) << 1;
				code |= (src.at<_Tp>(i, j - 1) >= center) << 0;
				dst.at<unsigned char>(i - 1, j - 1) = code;
			}
		}
	}

	// 外部接口，根据不同的数据类型调用模板函数
	void olbp(InputArray src, OutputArray dst) {
		switch (src.getMat().type()) {
		case CV_8SC1:   olbp_<char>(src, dst); break;
		case CV_8UC1:   olbp_<unsigned char>(src, dst); break;
		case CV_16SC1:  olbp_<short>(src, dst); break;
		case CV_16UC1:  olbp_<unsigned short>(src, dst); break;
		case CV_32SC1:  olbp_<int>(src, dst); break;
		case CV_32FC1:  olbp_<float>(src, dst); break;
		case CV_64FC1:  olbp_<double>(src, dst); break;
		default:
			string error_msg = format("Using Original Local Binary Patterns for feature extraction only works\n on single-channel images (given %d). Please pass the image data as a grayscale image!", src.getMat().type());
			CV_Error(CV_StsNotImplemented, error_msg);
			break;
		}
	}

	Mat olbp(InputArray src) {
		Mat dst;
		olbp(src, dst);
		return dst;
	}

	//--------------------------------------------------------
	// 计算LBP 直方图
	void My_calc_Hist(Mat& m, Mat& hist){

		/// 设定bin数目
		int histSize = 60;

		/// 设定取值范围 ( R,G,B) )
		float range[] = { 0, 255 };       //上下界区间
		const float* histRange = { range };

		bool uniform = true;
		bool accumulate = false;

		//储存直方图的矩阵
		Mat r_hist, g_hist, b_hist;

		/// 计算直方图:
		//&rgb_planes[0]: 输入数组(或数组集)
		//1: 输入数组的个数 (这里我们使用了一个单通道图像，我们也可以输入数组集 )
		//0: 需要统计的通道 (dim)索引 ，这里我们只是统计了灰度 (且每个数组都是单通道)所以只要写 0 就行了。
		//Mat(): 掩码( 0 表示忽略该像素)， 如果未定义，则不使用掩码
		//r_hist: 储存直方图的矩阵
		//1: 直方图维数
		//histSize: 每个维度的bin数目
		//histRange: 每个维度的取值范围
		//uniform 和 accumulate: bin大小相同，清楚直方图痕迹

		calcHist(&m, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	}

}