#include "usrGameController.h"
#include <queue>
#include <iostream>
#include <opencv2/nonfree/nonfree.hpp>//SIFT
#include <opencv2/legacy/legacy.hpp>//BFMatch暴力匹配
#include <math.h>
#ifdef VIA_OPENCV
//构造与初始化

bool click = false;

void maxLocs(const cv::Mat& src, std::queue<cv::Point>& dst, size_t size)
{
	float maxValue = -1.0f * std::numeric_limits<float>::max();
	float* srcData = reinterpret_cast<float*>(src.data);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (srcData[i*src.cols + j] > maxValue)
			{
				maxValue = srcData[i*src.cols + j];

				dst.push(cv::Point(j, i));

				// pop the smaller one off the end if we reach the size threshold.
				if (dst.size() > size)
				{
					dst.pop();
				}
			}
		}
	}
}

usrGameController::usrGameController(void* qtCD)
{
	qDebug() << "usrGameController online.";
	device = new deviceCyberDip(qtCD);//设备代理类
	cv::namedWindow(WIN_NAME);
	cv::setMouseCallback(WIN_NAME, mouseCallback, (void*)&(argM));

}

//析构
usrGameController::~usrGameController()
{
	cv::destroyAllWindows();
	if (device != nullptr)
	{
		delete device;
	}
	qDebug() << "usrGameController offline.";
}



//处理图像 
int usrGameController::usrProcessImage(cv::Mat& img)
{
	int board[4][4] = {};

	cv::Size imgSize(img.cols, img.rows - UP_CUT);
	if (imgSize.height <= 0 || imgSize.width <= 0)
	{
		qDebug() << "Invalid image. Size:" << imgSize.width << "x" << imgSize.height;
		return -1;
	}

	//截取图像边缘
	cv::Mat pt = img(cv::Rect(0, UP_CUT, imgSize.width, imgSize.height)); //pt = cropped image
	cv::imshow(WIN_NAME, pt);
	cv::imshow("wtf", pt);
	if (click){
		cv::Mat canny;
		/*cv::cvtColor(pt, bw, CV_BGR2GRAY);
		cv::blur(bw, bw, cv::Size(3, 3));*/
		cv::GaussianBlur(pt, canny, cv::Size(3, 3), 0);
		cv::Canny(pt, canny, 100, 100, 3);
		//imshow("canny", canny);

		//Perform Warping
		cv::vector<cv::vector<cv::Point>> contours;    //储存轮廓
		cv::vector<cv::Vec4i> hierarchy;

		findContours(canny, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);    //获取轮廓
		cv::Mat linePic = cv::Mat::zeros(canny.rows, canny.cols, CV_8UC3);
		for (int index = 0; index < contours.size(); index++){
			drawContours(linePic, contours, index, cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 1, 8/*, hierarchy*/);
		}

		cv::vector<cv::vector<cv::Point>> polyContours(contours.size());
		int maxArea = 0;
		for (int index = 0; index < contours.size(); index++){
			if (contourArea(contours[index]) > contourArea(contours[maxArea]))
				maxArea = index;
			approxPolyDP(contours[index], polyContours[index], 10, true);
		}

		cv::Mat polyPic = cv::Mat::zeros(pt.size(), CV_8UC3);
		drawContours(polyPic, polyContours, maxArea, cv::Scalar(0, 0, 255/*rand() & 255, rand() & 255, rand() & 255*/), 2);

		cv::vector<int>  hull;
		convexHull(polyContours[maxArea], hull, false);

		for (int i = 0; i < hull.size(); ++i){
			circle(polyPic, polyContours[maxArea][i], 10, cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
		}
		addWeighted(polyPic, 0.5, pt, 0.5, 0, pt);

		bool sorted = false;
		int n = 4;
		while (!sorted){
			for (int i = 1; i < n; i++){
				sorted = true;
				if (polyContours[maxArea][i - 1].x > polyContours[maxArea][i].x){
					std::swap(polyContours[maxArea][i - 1], polyContours[maxArea][i]);
					sorted = false;
				}
			}
			n--;
		}

		cv::Mat dst_warp, dst_warp2, dst_warpRotateScale, dst_warpTransformation, dst_warpFlip;
		dst_warp = pt(cv::Rect(0, UP_CUT, imgSize.width - 300, imgSize.height));
		cv::Point2f srcPoints[4], dstPoints[4];
		if (polyContours[maxArea][1].y < polyContours[maxArea][0].y){
			srcPoints[0] = polyContours[maxArea][1];
			srcPoints[2] = polyContours[maxArea][0];
		}
		else{
			srcPoints[0] = polyContours[maxArea][0];
			srcPoints[2] = polyContours[maxArea][1];
		}
		if (polyContours[maxArea][3].y < polyContours[maxArea][2].y){
			srcPoints[1] = polyContours[maxArea][3];
			srcPoints[3] = polyContours[maxArea][2];
		}
		else{
			srcPoints[1] = polyContours[maxArea][2];
			srcPoints[3] = polyContours[maxArea][3];
		}

		dstPoints[0] = cv::Point2f(0, 0);
		dstPoints[1] = cv::Point2f(canny.rows - 100, 0);  //bw.rows-200 -> shrink width
		dstPoints[2] = cv::Point2f(0, canny.cols - 200); //bw.cols-200 -> shrink height
		dstPoints[3] = cv::Point2f(canny.rows - 100, canny.cols - 200); //bw.rows-200 -> shrink width, bw.cols-200 -> shrink height

		cv::Mat M1 = cv::getPerspectiveTransform(srcPoints, dstPoints);//由四个点对计算透视变换矩阵  

		warpPerspective(pt, dst_warp, M1, dst_warp.size());//仿射变换  
		//cv::imshow("After_Warp", dst_warp);
		CvSVM svm;
		svm.load("SVM_HOG2.xml");
		
		int k = 0, l = 0;
		for (int i = 0; i < 4; ++i){
			for (int j = 0; j < 4; ++j){
				cv::Mat crop;
				if (j != 0)
				{
					k = 1;
				}
				else { k = 0; }
				if (i != 0)
				{
					l = 1;
				}
				else { l = 0; }
				/*	cv::circle(img,
				cv::Point(40 + 85 * (j)+k * 15 * j, 150 + 75 * (i)+l * 15 * i),
				5,
				cv::Scalar(250, 250, 50),
				2,
				8,
				0);*/

				//std::cout << 'x' << 45 + 90 * (j)+k * 15 * j << 'y' << 155 + 85 * (i)+l * 15 * i << std::endl;
				crop = dst_warp(cv::Rect(45 + 96 * (j)+k * 5 * j, 155 + 84 * (i)+l * 5 * i, 96, 84));
				cv::HOGDescriptor hog(cv::Size(32, 32), cv::Size(8, 8), cv::Size(4, 4), cv::Size(4, 4), 9);
				std::vector<float> descriptors;//HOG描述子向量
				hog.compute(crop, descriptors, cv::Size(4, 4));//计算HOG描述子，检测窗口移动步长(8,8)
				cv::Mat M2 = cv::Mat(descriptors.size(), 1, CV_32FC1);
				memcpy(M2.data, descriptors.data(), descriptors.size()*sizeof(float));
				int r = svm.predict(M2);
				board[i][j] = r;
			}
		}
		for (int k = 0; k < 4; ++k){
			for (int l = 0; l < 4; ++l){
				std::cout << '[' << board[l][k] << ']';
			}
			std::cout << std::endl;
		}	
			click = false;
		}


		///////////////////////////////////////////////////////////template matchin

		//cv::imshow("target", dst_warp);
		//cv::imshow("result", image_matched);
		
	/////////////////////////////////////////////////////////////


	//判断鼠标点击尺寸
	if (argM.box.x >= 0 && argM.box.x < imgSize.width&&
		argM.box.y >= 0 && argM.box.y < imgSize.height
		)
	{
		qDebug() << "\nX:" << argM.box.x << " Y:" << argM.box.y;
		if (argM.Hit)
		{
			device->comHitDown();
		}
		device->comMoveToScale(((double)argM.box.x + argM.box.width) / pt.cols, ((double)argM.box.y + argM.box.height) / pt.rows);
		argM.box.x = -1; argM.box.y = -1;
		if (argM.Hit)
		{
			device->comHitUp();
		}
		else
		{
			device->comHitOnce();
		}
	}
	return 0;
}

//鼠标回调函数
void mouseCallback(int event, int x, int y, int flags, void*param)
{
	usrGameController::MouseArgs* m_arg = (usrGameController::MouseArgs*)param;
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE: // 鼠标移动时
	{
								 if (m_arg->Drawing)
								 {
									 m_arg->box.width = x - m_arg->box.x;
									 m_arg->box.height = y - m_arg->box.y;
								 }
	}
		break;
	case CV_EVENT_LBUTTONDOWN:case CV_EVENT_RBUTTONDOWN: // 左/右键按下
	{
								  m_arg->Hit = event == CV_EVENT_RBUTTONDOWN;
								  m_arg->Drawing = true;
								  m_arg->box = cvRect(x, y, 0, 0);
								  click = true;
	}
		break;
	case CV_EVENT_LBUTTONUP:case CV_EVENT_RBUTTONUP: // 左/右键弹起
	{
								m_arg->Hit = false;
								m_arg->Drawing = false;
								if (m_arg->box.width < 0)
								{
									m_arg->box.x += m_arg->box.width;
									m_arg->box.width *= -1;
								}
								if (m_arg->box.height < 0)
								{
									m_arg->box.y += m_arg->box.height;
									m_arg->box.height *= -1;
								}
	}
		break;
	}
}
#endif
