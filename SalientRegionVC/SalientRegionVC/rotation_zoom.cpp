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
int main()
{
Mat srcImg = imread("DesertSafari.jpg");//https://blog.csdn.net/wangyaninglm/article/details/44020489测试集
Mat dstImg = imread("DesertSafari.jpg");

//在原图上加mask
Mat segImg;
srcImg.copyTo(segImg, dst);
imshow("分割后的原图", segImg);
finish = clock();
MeanshiftTime = (double)(finish - start) / CLOCKS_PER_SEC;
cout << "\t\t\t" << "分割原图耗时:" << MeanshiftTime << endl;
waitKey(0);

SiftFeatureDetector detector;// (800);        //定义特点点检测器 海塞矩阵阈值800
vector<KeyPoint> keypoint01, keypoint02;//定义两个容器存放特征点
detector.detect(segImg, keypoint01);
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
drawKeypoints(segImg, keypoint01, out_srcImg);
drawKeypoints(dstImg, keypoint02, out_dstImg);
imshow("特征点图01", out_srcImg);
imshow("特征点图02", out_dstImg);

//提取特征点的特征向量（128维）
SiftDescriptorExtractor extractor;
Mat descriptor01, descriptor02;
/*extractor.compute(srcImg, keypoint01, descriptor01);
extractor.compute(dstImg, keypoint02, descriptor02);*/
extractor.compute(segImg, keypoint01, descriptor01);
extractor.compute(dstImg, keypoint02, descriptor02);

//匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配

//BruteForceMatcher<L2<float>> matcher;
//vector<DMatch> matches;
FlannBasedMatcher matcher;
vector<DMatch> matches;
Mat img_matches;
matcher.match(descriptor01, descriptor02, matches);
drawMatches(segImg, keypoint01, dstImg, keypoint02, matches, img_matches);
imshow("误匹配消除前", img_matches);
sort(matches.begin(), matches.end()); //特征点排序，opencv按照匹配点准确度排序      
//获取排在前N个的最优匹配特征点    
vector<Point2f> imagePoints1, imagePoints2;
for (int i = 0; i < 10; i++)
{
	imagePoints1.push_back(keypoint01[matches[i].queryIdx].pt);
	imagePoints2.push_back(keypoint02[matches[i].trainIdx].pt);
}

//获取图像1到图像2的投影映射矩阵，尺寸为3*3    
Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
//Mat img1 = imread("beaver.png");
//Mat img5 = imread("beaver_xform.png");
Mat warp = dstImg.clone();
warpPerspective(srcImg, warp, homo, cv::Size(warp.cols, warp.rows), 1, 0, cv::Scalar(255, 255, 255));//这里的size要cv::size,而不能是Mat::size
imshow("配准图像", warp);
finish = clock();
totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
cout << "\t\t\t" << "总耗时:" << totaltime << endl;
double psnr = PSNR(dstImg, warp);
cout << "\t\t\t" << "峰值信噪比：" << psnr << endl;
waitKey(0);
return 1;
}