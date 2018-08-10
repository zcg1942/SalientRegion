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
void FloodFillProcess(Mat &Sal1);
#define LODIFF (6)//宏定义最好用大写字母，和正规代码分开，并且不能有分号
#define UPDIFF (6)
//http://ivrlwww.epfl.ch/~achanta/SalientRegionDetection/SalientRegionDetection.html
//http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/
int main(void)
{
	clock_t start, finish,salstart;
	double totaltime,salTime,MeanshiftTime;
	//读取图像
	start = clock();
	Mat srcImg = imread("graf1.ppm");//https://blog.csdn.net/wangyaninglm/article/details/44020489测试集
	//求出原图均值和方差
	Mat tmp_m, tmp_sd;
	double m = 0, sd = 0;
	double mb = 0, mg = 0, mr = 0;

	mb = mean(srcImg)[0];
	mg = mean(srcImg)[1];//求三个通道的均值
	mr = mean(srcImg)[2];
	cout << "Mean: " << mb <<" "<<mg<<" "<<mr<<endl;

	
	meanStdDev(srcImg, tmp_m, tmp_sd);
	m = tmp_m.at<double>(0, 0);
	sd = tmp_sd.at<double>(0, 0);
	cout << "Mean: " << m << " , StdDev: " << sd << endl;

	Mat ground((srcImg.rows) * 2, (srcImg.cols) * 2, CV_8UC3, Scalar::all(m));
	//Mat ground((srcImg.rows) * 2, (srcImg.cols) * 2, CV_8UC3, Scalar(mb,mg,mr));//彩色填充 因为FT方法要利用亮度和颜色信息 所以这里通道数要改为3 scalar也填三个通道
	Mat ROI = ground(Rect(0, 0, srcImg.cols, srcImg.rows));
	//Mat mask=
	srcImg.copyTo(ROI);//copy当然只能复制到相同大小的图像中，只不过第一个参数可以是另一幅图的指定的ROI
	//Mat dstImg = imread("DesertSafari.jpg");
	Mat res1,res2;// = srcImg.clone();
	if (srcImg.empty())
	{
		cout << "图像没有读取成功" << endl;
		getchar();
		return 0;
	}
	imshow("原图1", srcImg);
	imshow("放大背景", ground);
	imwrite("1000dx.png", ground);
	//waitKey(0);
	Mat Sal1 = Mat::zeros(ground.size(), CV_8UC1);//Mat初始化的方法要会
	//Mat Sal2 = Mat::zeros(dstImg.size(), CV_8UC1);

	Sal1 = SalientRegionDetectionBasedonFT(ground, Sal1);
	//Sal2=SalientRegionDetectionBasedonFT(dstImg,Sal2);

	imshow("salmap1", Sal1);
	imwrite("000.png", Sal1);

	////meanshift floodfill系列
	////imshow("原图2", dstImg);
	//int spatialRad, colorRad, maxPyrLevel;
	//spatialRad = 35;//空间域半径
	//colorRad = 40;//颜色域半径
	//maxPyrLevel = 3;//金字塔最大层数 
	////https://blog.csdn.net/gdfsg/article/details/50975422
	//pyrMeanShiftFiltering(srcImg, res1, spatialRad, colorRad, maxPyrLevel);
	////pyrMeanShiftFiltering(dstImg, res2, spatialRad, colorRad, maxPyrLevel);
	//imshow("MeanshiftSeg1", res1);
	////imshow("MeanshiftSeg2", res2);
	//finish = clock();
	//salstart = clock();
	//MeanshiftTime = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "\t\t\t" << "均值漂移耗时:" << MeanshiftTime << endl;

	//


	
	////imshow("salmap2", Sal2);
	//finish = clock();
	//salTime = (double)(finish - salstart) / CLOCKS_PER_SEC;
	//cout << "\t\t\t" << "显著性区域提取耗时:" << salTime << endl;

	////显著性提取之后漫水填充
	////FloodFillProcess(Sal1);
	//RNG rng = theRNG();
	//Mat mask(Sal1.rows + 2, Sal1.cols + 2, CV_8UC1, Scalar::all(0));  //掩模  
	//for (int y = 0; y < Sal1.rows; y++)
	//{
	//	for (int x = 0; x < Sal1.cols; x++)
	//	{
	//		if (mask.at<uchar>(y + 1, x + 1) == 0)  //非0处即为1，表示已经经过填充，不再处理  
	//		{
	//			Scalar newVal(rng(256), rng(256), rng(256));
	//			floodFill(Sal1, mask, Point(x, y), newVal, 0, Scalar::all(LODIFF), Scalar::all(UPDIFF)); //执行漫水填充  
	//		}//参数很重要，将来可以用机器学习的改进
	//	}
	//}
	//finish = clock();
	//salTime = (double)(finish - salstart) / CLOCKS_PER_SEC;
	//cout << "\t\t\t" << "漫水填充耗时:" << salTime << endl;
	////Mat Sal3 = Mat::zeros(Sal1.size(), CV_8UC1);

	////SalientRegionDetectionBasedonFT(Sal1, Sal3);//函数调用另外一个函数的问题
	//imshow("meanShift图像分割", Sal1);
	////imwrite("Fill.png",Sal1);//后缀后缀！！！
	//Mat dilateforFlood = Mat::zeros(Sal1.size(), CV_8UC1);
	//int type=Sal1.type();//发现type=0，单通道灰度图
	///*Mat flood = Sal1.clone();
	//imshow("hhg",flood);*/
	////填充之后再进行显著性检测 感觉没什么用
	////Mat Sal3 = Sal1.clone();;
	////Sal3= SalientRegionDetectionBasedonFT(Sal1, Sal3);
	////imshow("hh", Sal3);
	////imwrite("Salforfill96.png", Sal3);


	//cv::Mat element(2, 2, CV_8U, cv::Scalar(255));
	////dilate(Sal1, dilateforFlood, element);
	////腐蚀 并进行显著性检测
	//erode(Sal1, dilateforFlood, element);
	//cv::imshow("dilate Image", dilateforFlood);
	////imwrite("erodefill96.png", dilateforFlood);
	//Mat  FloodFuSal = dilateforFlood.clone();
	//SalientRegionDetectionBasedonFT(dilateforFlood, FloodFuSal);
	//cv::imshow("dilate Sal Image", FloodFuSal);
	//imwrite("Salforerode106beaver.png", FloodFuSal);

	//finish = clock();
	//salTime = (double)(finish - salstart) / CLOCKS_PER_SEC;
	//cout << "\t\t\t" << "填充腐蚀显著性耗时:" << salTime << endl;
	waitKey(0);





	//imwrite("sal1.png", Sal1);//注意要加格式后缀
	//imwrite("sal2.png", Sal2);//保存显著性检测后的图像
	//SegToBin(Sal1);//这样二值化后的图像全黑，导致后面检测不到特征点而出错
	//SegToBin(Sal2);//应该是写的二值化函数不对，也应该在lab空间内计算均值
	//imshow("二值化1", Sal1);
	//imshow("二值化2", Sal2);
	
	
	
}

//https://blog.csdn.net/cai13160674275/article/details/72991049
//http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/
Mat SalientRegionDetectionBasedonFT(Mat &src,Mat &Sal){
	Mat Lab,BGR;
	if (src.type()==16)
	cvtColor(src, Lab, CV_BGR2Lab);//第一个参数是三通道的，而在显著性提取之后是单通道
	else
	{
		cvtColor(src, BGR, CV_GRAY2BGR);//如果是灰度图，先转换为三色图
		cvtColor(BGR, Lab, CV_BGR2Lab);
	}


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

	cout << "\t\t\t" << "像素显著性最值:" << max_v << " " << min_v << endl;//输出最大值和最小值
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
//将图一背景扩大化，这样显著性检测和均值漂移就可以更好地突出主体
//一开始想按像素计算，这样麻烦了，想到用研磨的方法
void AddGround(Mat src, Mat dst)
{

}
//漫水填充
void FloodFillProcess(Mat& Sal1)
{
	RNG rng = theRNG();
	Mat mask(Sal1.rows + 2, Sal1.cols + 2, CV_8UC1, Scalar::all(0));  //掩模  
	for (int y = 0; y < Sal1.rows; y++)
	{
		for (int x = 0; x < Sal1.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)  //非0处即为1，表示已经经过填充，不再处理  
			{
				Scalar newVal(rng(256), rng(256), rng(256));
				floodFill(Sal1, mask, Point(x, y), newVal, 0, Scalar::all(0.51), Scalar::all(0.5)); //执行漫水填充  
			}//参数很重要，将来可以用机器学习的改进
		}
	}
	
	//Mat Sal3 = Mat::zeros(Sal1.size(), CV_8UC1);

	//SalientRegionDetectionBasedonFT(Sal1, Sal3);//函数调用另外一个函数的问题
	imshow("meanShift图像分割", Sal1);

	waitKey(0);
}