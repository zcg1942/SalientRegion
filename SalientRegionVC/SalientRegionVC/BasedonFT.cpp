#include<stdio.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<time.h>
using namespace std;
using namespace cv;
Mat SalientRegionDetectionBasedonFT(Mat &src,Mat &Sal);
Mat SegToBin(Mat &src);
int main(void)
{
	clock_t start, finish;
	double totaltime,salTime;
	//��ȡͼ��
	start = clock();
	Mat srcImg = imread("img1_salientobject.jpg");//https://blog.csdn.net/wangyaninglm/article/details/44020489���Լ�
	Mat dstImg = imread("img5_salientobject.jpg");
	Mat res1,res2;// = srcImg.clone();
	if (srcImg.empty() || dstImg.empty())
	{
		cout << "ͼ��û�ж�ȡ�ɹ�" << endl;
		return 0;
	}
	imshow("ԭͼ1", srcImg);
	imshow("ԭͼ2", dstImg);
	int spatialRad, colorRad, maxPyrLevel;
	spatialRad = 35;//�ռ���뾶
	colorRad = 40;//��ɫ��뾶
	maxPyrLevel = 3;//������������ 
	//https://blog.csdn.net/gdfsg/article/details/50975422
	/*pyrMeanShiftFiltering(srcImg, res1, spatialRad, colorRad, maxPyrLevel);
	pyrMeanShiftFiltering(dstImg, res2, spatialRad, colorRad, maxPyrLevel);
	imshow("MeanshiftSeg1", res1);
	imshow("MeanshiftSeg2", res2);*/

	Mat Sal1 = Mat::zeros(srcImg.size(), CV_8UC1);
	Mat Sal2 = Mat::zeros(dstImg.size(), CV_8UC1);

	///*Sal1=SalientRegionDetectionBasedonFT(srcImg,Sal1);
	//Sal2=SalientRegionDetectionBasedonFT(dstImg,Sal2);
	//finish = clock();
	//salTime = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "������������ȡ��ʱ:" <<salTime<< endl;
	//imshow("salmap1", Sal1);
	//imshow("salmap2", Sal2);*/

	//imwrite("sal1.png", Sal1);//ע��Ҫ�Ӹ�ʽ��׺
	//imwrite("sal2.png", Sal2);//���������Լ����ͼ��
	//SegToBin(Sal1);//������ֵ�����ͼ��ȫ�ڣ����º����ⲻ�������������
	//SegToBin(Sal2);//Ӧ����д�Ķ�ֵ���������ԣ�ҲӦ����lab�ռ��ڼ����ֵ
	//imshow("��ֵ��1", Sal1);
	//imshow("��ֵ��2", Sal2);
	
	
	//SIFT�������
	SiftFeatureDetector detector;// (800);        //�����ص������ ����������ֵ800
	vector<KeyPoint> keypoint01, keypoint02;//���������������������
	detector.detect(srcImg, keypoint01);
	detector.detect(dstImg, keypoint02);
	int n1 = keypoint01.size();
	int n2 = keypoint02.size();
	//detector.detect(srcImg, keypoint01);
	//detector.detect(dstImg, keypoint02);
	//int n1 = keypoint01.size();//���Ų�����
	//int n2 = keypoint02.size();

	//������ͼ�л�����⵽��������
	Mat out_srcImg;
	Mat out_dstImg;
	drawKeypoints(srcImg, keypoint01, out_srcImg);
	drawKeypoints(dstImg, keypoint02, out_dstImg);
	imshow("������ͼ01", out_srcImg);
	imshow("������ͼ02", out_dstImg);

	//��ȡ�����������������128ά��
	SiftDescriptorExtractor extractor;
	Mat descriptor01, descriptor02;
	/*extractor.compute(srcImg, keypoint01, descriptor01);
	extractor.compute(dstImg, keypoint02, descriptor02);*/
	extractor.compute(srcImg, keypoint01, descriptor01);
	extractor.compute(dstImg, keypoint02, descriptor02);

	//ƥ�������㣬��Ҫ������������������������ŷʽ���룬����С��ĳ����ֵ����Ϊƥ��

	//BruteForceMatcher<L2<float>> matcher;
	//vector<DMatch> matches;
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	Mat img_matches;
	matcher.match(descriptor01, descriptor02, matches);
	drawMatches(srcImg, keypoint01, dstImg, keypoint02, matches, img_matches);
	imshow("��ƥ������ǰ", img_matches);
	sort(matches.begin(), matches.end()); //����������opencv����ƥ���׼ȷ������      
	//��ȡ����ǰN��������ƥ��������    
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i<10; i++)
	{
		imagePoints1.push_back(keypoint01[matches[i].queryIdx].pt);
		imagePoints2.push_back(keypoint02[matches[i].trainIdx].pt);
	}

	//��ȡͼ��1��ͼ��2��ͶӰӳ����󣬳ߴ�Ϊ3*3    
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	Mat img1 = imread("img1.jpg");
	Mat img5 = imread("img5.jpg");
	Mat warp = img5.clone();
	warpPerspective(img1, img5, homo, cv::Size(warp.cols, warp.rows),1,0,cv::Scalar(255,255,255));//�����sizeҪcv::size,��������Mat::size
	imshow("��׼ͼ��", warp);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "��ʱ:" << totaltime << endl;
	double psnr=PSNR(img5, warp);
	cout << "��ֵ����ȣ�" << psnr<<endl;
	waitKey(0);
	return 1;
}

//https://blog.csdn.net/cai13160674275/article/details/72991049
//http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/
Mat SalientRegionDetectionBasedonFT(Mat &src,Mat &Sal){
	Mat Lab;
	cvtColor(src, Lab, CV_BGR2Lab);

	int row = src.rows, col = src.cols;

	//int Sal_org[row][col];
	int **Sal_org;//https://zhidao.baidu.com/question/462803761.html ����ָ��ʵ�������С�ñ�������
	Sal_org = new int*[row]; 
	for (int i = 0; i < row; i++)
		Sal_org[i] = new int[col];
	//memset(Sal_org, 0, sizeof(Sal_org));

	Point3_<uchar>* p;

	int MeanL = 0, Meana = 0, Meanb = 0;
	for (int i = 0; i<row; i++){
		for (int j = 0; j<col; j++){
			p = Lab.ptr<Point3_<uchar> >(i, j);
			MeanL += p->x;
			Meana += p->y;
			Meanb += p->z;
		}
	}
	MeanL /= (row*col);//ƽ��ֵ
	Meana /= (row*col);
	Meanb /= (row*col);

	GaussianBlur(Lab, Lab, Size(3, 3), 0, 0);

	

	int val;

	int max_v = 0;
	int min_v = 1 << 28;//???

	for (int i = 0; i<row; i++){
		for (int j = 0; j<col; j++){
			p = Lab.ptr<Point3_<uchar> >(i, j);
			val = sqrt((MeanL - p->x)*(MeanL - p->x) + (p->y - Meana)*(p->y - Meana) + (p->z - Meanb)*(p->z - Meanb));//lab�ռ�ľ�ֵ��ȥ��ǰ����ֵ ����ÿһ�����ص�������
			Sal_org[i][j] = val;
			max_v = max(max_v, val);//����������֮��ϴ��
			min_v = min(min_v, val);
		}
	}

	cout <<"������������ֵ:"<< max_v << " " << min_v << endl;//������ֵ����Сֵ
	int X, Y, Mean_sal = 0;
	for (Y = 0; Y < row; Y++)
	{
		for (X = 0; X < col; X++)
		{
			Sal.at<uchar>(Y, X) = (Sal_org[Y][X] - min_v) * 255 / (max_v - min_v);        //    ����ȫͼÿ�����ص������� ��һ����0~255�ĻҶ�ֵ
			//Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    ����ȫͼÿ�����ص�������
			//Mean_sal += Sal.at<uchar>(Y, X);
		
		}
	}
	return Sal;
	//imshow("sal", Sal);
	//waitKey(0);
}
Mat SegToBin(Mat &src)
{
	int row = src.rows, col = src.cols;
	int Mean_sal=0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			Mean_sal += src.at<uchar>(i, j);
		}
	}
	Mean_sal = Mean_sal / (row*col);

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (src.at<uchar>(i, j)>Mean_sal)//��ֵ�Ǿ�ֵ��2��
				src.at<uchar>(i, j) = 255;
			else src.at<uchar>(i, j) = 0;
		}
	}
	return src;

}
//static void meanShiftSegmentation(int, void*)
//{
//	cout << "spatialRad=" << spatialRad << "; "
//		<< "colorRad=" << colorRad << "; "
//		<< "maxPyrLevel=" << maxPyrLevel << endl;
//	pyrMeanShiftFiltering(img, res, spatialRad, colorRad, maxPyrLevel);
//	 
//	//floodFillPostprocess(res, Scalar::all(2));
//	imshow(winName, res);
//}