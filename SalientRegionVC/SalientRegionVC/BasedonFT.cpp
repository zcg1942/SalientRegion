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
	//读取图像
	start = clock();
	Mat srcImg = imread("img1_salientobject.jpg");//https://blog.csdn.net/wangyaninglm/article/details/44020489测试集
	Mat dstImg = imread("img5_salientobject.jpg");
	Mat res1,res2;// = srcImg.clone();
	if (srcImg.empty() || dstImg.empty())
	{
		cout << "图像没有读取成功" << endl;
		return 0;
	}
	imshow("原图1", srcImg);
	imshow("原图2", dstImg);
	int spatialRad, colorRad, maxPyrLevel;
	spatialRad = 35;//空间域半径
	colorRad = 40;//颜色域半径
	maxPyrLevel = 3;//金字塔最大层数 
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
	//cout << "显著性区域提取耗时:" <<salTime<< endl;
	//imshow("salmap1", Sal1);
	//imshow("salmap2", Sal2);*/

	//imwrite("sal1.png", Sal1);//注意要加格式后缀
	//imwrite("sal2.png", Sal2);//保存显著性检测后的图像
	//SegToBin(Sal1);//这样二值化后的图像全黑，导致后面检测不到特征点而出错
	//SegToBin(Sal2);//应该是写的二值化函数不对，也应该在lab空间内计算均值
	//imshow("二值化1", Sal1);
	//imshow("二值化2", Sal2);
	
	
	//SIFT特征检测
	SiftFeatureDetector detector;// (800);        //定义特点点检测器 海塞矩阵阈值800
	vector<KeyPoint> keypoint01, keypoint02;//定义两个容器存放特征点
	detector.detect(srcImg, keypoint01);
	detector.detect(dstImg, keypoint02);
	int n1 = keypoint01.size();
	int n2 = keypoint02.size();
	//detector.detect(srcImg, keypoint01);
	//detector.detect(dstImg, keypoint02);
	//int n1 = keypoint01.size();//括号不能少
	//int n2 = keypoint02.size();

	//在两幅图中画出检测到的特征点
	Mat out_srcImg;
	Mat out_dstImg;
	drawKeypoints(srcImg, keypoint01, out_srcImg);
	drawKeypoints(dstImg, keypoint02, out_dstImg);
	imshow("特征点图01", out_srcImg);
	imshow("特征点图02", out_dstImg);

	//提取特征点的特征向量（128维）
	SiftDescriptorExtractor extractor;
	Mat descriptor01, descriptor02;
	/*extractor.compute(srcImg, keypoint01, descriptor01);
	extractor.compute(dstImg, keypoint02, descriptor02);*/
	extractor.compute(srcImg, keypoint01, descriptor01);
	extractor.compute(dstImg, keypoint02, descriptor02);

	//匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配

	//BruteForceMatcher<L2<float>> matcher;
	//vector<DMatch> matches;
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	Mat img_matches;
	matcher.match(descriptor01, descriptor02, matches);
	drawMatches(srcImg, keypoint01, dstImg, keypoint02, matches, img_matches);
	imshow("误匹配消除前", img_matches);
	sort(matches.begin(), matches.end()); //特征点排序，opencv按照匹配点准确度排序      
	//获取排在前N个的最优匹配特征点    
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i<10; i++)
	{
		imagePoints1.push_back(keypoint01[matches[i].queryIdx].pt);
		imagePoints2.push_back(keypoint02[matches[i].trainIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵，尺寸为3*3    
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	Mat img1 = imread("img1.jpg");
	Mat img5 = imread("img5.jpg");
	Mat warp = img5.clone();
	warpPerspective(img1, img5, homo, cv::Size(warp.cols, warp.rows),1,0,cv::Scalar(255,255,255));//这里的size要cv::size,而不能是Mat::size
	imshow("配准图像", warp);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "耗时:" << totaltime << endl;
	double psnr=PSNR(img5, warp);
	cout << "峰值信噪比：" << psnr<<endl;
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
	int **Sal_org;//https://zhidao.baidu.com/question/462803761.html 二级指针实现数组大小用变量定义
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
	MeanL /= (row*col);//平均值
	Meana /= (row*col);
	Meanb /= (row*col);

	GaussianBlur(Lab, Lab, Size(3, 3), 0, 0);

	

	int val;

	int max_v = 0;
	int min_v = 1 << 28;//???

	for (int i = 0; i<row; i++){
		for (int j = 0; j<col; j++){
			p = Lab.ptr<Point3_<uchar> >(i, j);
			val = sqrt((MeanL - p->x)*(MeanL - p->x) + (p->y - Meana)*(p->y - Meana) + (p->z - Meanb)*(p->z - Meanb));//lab空间的均值减去当前像素值 计算每一个像素的显著性
			Sal_org[i][j] = val;
			max_v = max(max_v, val);//返回两个数之间较大的
			min_v = min(min_v, val);
		}
	}

	cout <<"像素显著性最值:"<< max_v << " " << min_v << endl;//输出最大值和最小值
	int X, Y, Mean_sal = 0;
	for (Y = 0; Y < row; Y++)
	{
		for (X = 0; X < col; X++)
		{
			Sal.at<uchar>(Y, X) = (Sal_org[Y][X] - min_v) * 255 / (max_v - min_v);        //    计算全图每个像素的显著性 归一化到0~255的灰度值
			//Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
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
			if (src.at<uchar>(i, j)>Mean_sal)//阈值是均值的2倍
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