#include <cv.h>
#include <opencv2/opencv.hpp>  
#define cvQueryHistValue_1D( hist, idx0 ) \
    ((float)cvGetReal1D( (hist)->bins, (idx0)))
using namespace cv;

float calc_entropy(CvHistogram *hist, int begin, int end)
{
	float total = 0;  // 总概率
	// 得到总的Pi
	for (int i = begin; i < end; i++)
	{
		total += cvQueryHistValue_1D(hist, i);
	}

	float entropy = 0;  // 熵

	for (int i = begin; i < end; i++)
	{
		float probability = cvQueryHistValue_1D(hist, i);
		if (probability == 0)
			continue;
		probability /= total;

		entropy += -probability * log(probability);
	}

	return entropy;
}

int ksw_entropy(IplImage *img)
{
	assert(img != NULL);
	assert(img->depth == 8);
	assert(img->nChannels == 1);

	float range[2] = { 0,255 };
	float *ranges[1] = { &range[0] };
	int sizes = 256;

	// 创建直方图
	CvHistogram *hist = cvCreateHist(1, &sizes, CV_HIST_ARRAY, ranges, 1);
	// 直方图计算
	cvCalcHist(&img, hist, 0, 0);
	// 直方图归一化
	cvNormalizeHist(hist, 1.0);

	int threshold = 0;
	float max_entropy = 0;
	// 循环计算，得到做大熵以及分割阈值
	for (int i = 0; i < sizes; i++)
	{
		float entropy = calc_entropy(hist, 0, i) + calc_entropy(hist, i + 1, sizes);
		if (entropy > max_entropy)
		{
			max_entropy = entropy;
			threshold = i;
		}
	}
	return threshold;
}
int main(int argc, char **argv)
{
	IplImage *img = cvLoadImage("C:\\Users\\jmuaia007\\Pictures\\Saved Pictures\\jida2.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *reimg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

	int threshold = ksw_entropy(img);
	cvThreshold(img, reimg, threshold, 255, CV_THRESH_BINARY);

	cvNamedWindow("img");
	cvShowImage("img", img);
	cvNamedWindow("reimg");
	cvShowImage("reimg", reimg);

	cvWaitKey(0);
	return 0;
}
