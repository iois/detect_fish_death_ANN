#pragma once
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

namespace cv{

	//---------------------------����LBP----------------------
	// a Բ��LBP����

	// srcΪ����ͼ��dstΪ���ͼ��radiusΪ�뾶��neighborΪ���㵱ǰ��LBP������������ص�����Ҳ�������������
	template <typename _Tp> static // ģ�庯�������ݲ�ͬ��ԭʼ�������͵õ���ͬ�Ľ��
		inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors)
	{
		//get matrices
		Mat src = _src.getMat();
		// allocate memory for result��˲������ⲿ��_dst�����ڴ�ռ䣬����������Ͷ���int
		_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
		Mat dst = _dst.getMat();
		// zero
		dst.setTo(0);
		for (int n = 0; n < neighbors; n++)
		{
			// sample points ��ȡ��ǰ������
			float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n / static_cast<float>(neighbors));
			float y = static_cast<float>(radius)* cos(2.0*CV_PI*n / static_cast<float>(neighbors));
			// relative indices ��ȡ������ȡ��
			int fx = static_cast<int>(floor(x)); // ����ȡ��
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));  // ����ȡ��
			int cy = static_cast<int>(ceil(y));
			// fractional part С������
			float tx = x - fx;
			float ty = y - fy;
			// set interpolation weights �����ĸ���Ĳ�ֵȨ��
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 = tx  *      ty;
			// iterate through your data ѭ������ͼ������
			for (int i = radius; i < src.rows - radius; i++)
			{
				for (int j = radius; j < src.cols - radius; j++)
				{
					// calculate interpolated value �����ֵ��t��ʾ�ĸ����Ȩ�غ�
					float t = w1*src.at<_Tp>(i + fy, j + fx) +

						w2*src.at<_Tp>(i + fy, j + cx) +

						w3*src.at<_Tp>(i + cy, j + fx) +

						w4*src.at<_Tp>(i + cy, j + cx);

					// floating point precision, so check some machine-dependent epsilon
					// std::numeric_limits<float>::epsilon()=1.192092896e-07F
					// ��t>=src(i,j)��ʱ��ȡ1����������Ӧ����λ
					dst.at<int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) ||
						(std::abs(t - src.at<_Tp>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
				}
			}
		}
	}

	// �ⲿ�ӿڣ����ݲ�ͬ���������͵���ģ�庯��
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

	//b ԭʼLBP����

	// ԭʼLBP����ֻ�Ǽ���8�����ڵľֲ���ֵģʽ
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

	// �ⲿ�ӿڣ����ݲ�ͬ���������͵���ģ�庯��
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
	// ����LBP ֱ��ͼ
	void My_calc_Hist(Mat& m, Mat& hist){

		/// �趨bin��Ŀ
		int histSize = 60;

		/// �趨ȡֵ��Χ ( R,G,B) )
		float range[] = { 0, 255 };       //���½�����
		const float* histRange = { range };

		bool uniform = true;
		bool accumulate = false;

		//����ֱ��ͼ�ľ���
		Mat r_hist, g_hist, b_hist;

		/// ����ֱ��ͼ:
		//&rgb_planes[0]: ��������(�����鼯)
		//1: ��������ĸ��� (��������ʹ����һ����ͨ��ͼ������Ҳ�����������鼯 )
		//0: ��Ҫͳ�Ƶ�ͨ�� (dim)���� ����������ֻ��ͳ���˻Ҷ� (��ÿ�����鶼�ǵ�ͨ��)����ֻҪд 0 �����ˡ�
		//Mat(): ����( 0 ��ʾ���Ը�����)�� ���δ���壬��ʹ������
		//r_hist: ����ֱ��ͼ�ľ���
		//1: ֱ��ͼά��
		//histSize: ÿ��ά�ȵ�bin��Ŀ
		//histRange: ÿ��ά�ȵ�ȡֵ��Χ
		//uniform �� accumulate: bin��С��ͬ�����ֱ��ͼ�ۼ�

		calcHist(&m, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	}

}