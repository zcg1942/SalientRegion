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
Mat srcImg = imread("DesertSafari.jpg");//https://blog.csdn.net/wangyaninglm/article/details/44020489���Լ�
Mat dstImg = imread("DesertSafari.jpg");

//��ԭͼ�ϼ�mask
Mat segImg;
srcImg.copyTo(segImg, dst);
imshow("�ָ���ԭͼ", segImg);
finish = clock();
MeanshiftTime = (double)(finish - start) / CLOCKS_PER_SEC;
cout << "\t\t\t" << "�ָ�ԭͼ��ʱ:" << MeanshiftTime << endl;
waitKey(0);

SiftFeatureDetector detector;// (800);        //�����ص������ ����������ֵ800
vector<KeyPoint> keypoint01, keypoint02;//���������������������
detector.detect(segImg, keypoint01);
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
drawKeypoints(segImg, keypoint01, out_srcImg);
drawKeypoints(dstImg, keypoint02, out_dstImg);
imshow("������ͼ01", out_srcImg);
imshow("������ͼ02", out_dstImg);

//��ȡ�����������������128ά��
SiftDescriptorExtractor extractor;
Mat descriptor01, descriptor02;
/*extractor.compute(srcImg, keypoint01, descriptor01);
extractor.compute(dstImg, keypoint02, descriptor02);*/
extractor.compute(segImg, keypoint01, descriptor01);
extractor.compute(dstImg, keypoint02, descriptor02);

//ƥ�������㣬��Ҫ������������������������ŷʽ���룬����С��ĳ����ֵ����Ϊƥ��

//BruteForceMatcher<L2<float>> matcher;
//vector<DMatch> matches;
FlannBasedMatcher matcher;
vector<DMatch> matches;
Mat img_matches;
matcher.match(descriptor01, descriptor02, matches);
drawMatches(segImg, keypoint01, dstImg, keypoint02, matches, img_matches);
imshow("��ƥ������ǰ", img_matches);
sort(matches.begin(), matches.end()); //����������opencv����ƥ���׼ȷ������      
//��ȡ����ǰN��������ƥ��������    
vector<Point2f> imagePoints1, imagePoints2;
for (int i = 0; i < 10; i++)
{
	imagePoints1.push_back(keypoint01[matches[i].queryIdx].pt);
	imagePoints2.push_back(keypoint02[matches[i].trainIdx].pt);
}

//��ȡͼ��1��ͼ��2��ͶӰӳ����󣬳ߴ�Ϊ3*3    
Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
//Mat img1 = imread("beaver.png");
//Mat img5 = imread("beaver_xform.png");
Mat warp = dstImg.clone();
warpPerspective(srcImg, warp, homo, cv::Size(warp.cols, warp.rows), 1, 0, cv::Scalar(255, 255, 255));//�����sizeҪcv::size,��������Mat::size
imshow("��׼ͼ��", warp);
finish = clock();
totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
cout << "\t\t\t" << "�ܺ�ʱ:" << totaltime << endl;
double psnr = PSNR(dstImg, warp);
cout << "\t\t\t" << "��ֵ����ȣ�" << psnr << endl;
waitKey(0);
return 1;
}