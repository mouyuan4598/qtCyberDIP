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

		///*for (int i = 0; i < 4; ++i)
		//std::cout << polyContours[maxArea][i] << std::endl;*/
		//std::cout << pt.cols << ' ' << pt.rows;
		///*cv::GaussianBlur(dst_warp, dst_warp, cv::Size(3, 3), 0);
		//cv::Canny(dst_warp, dst_warp, 50, 50, 3);*/
		////////////////////////////CLAHE////////////////////////////////////
		////Perform CLAHE
		//cv::Mat lab_image;
		//cv::cvtColor(dst_warp, lab_image, CV_BGR2HSV);
		//// Extract the L channel
		//std::vector<cv::Mat> lab_planes(3);
		//cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]
		//// apply the CLAHE algorithm to the L channel
		//cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		//clahe->setClipLimit(4);
		//cv::Mat dst;
		//clahe->apply(lab_planes[2], dst);
		//// Merge the the color planes back into an Lab image
		//dst.copyTo(lab_planes[2]);
		//cv::merge(lab_planes, lab_image);
		//// convert back to RGB
		//cv::Mat image_clahe;
		//cv::cvtColor(lab_image, image_clahe, CV_HSV2BGR);
		//cv::imshow("After_CLAHE", image_clahe);

		cv::Mat his;
		cv::Mat dst_warp_RGB[3];
		cv::split(dst_warp, dst_warp_RGB);
		for (int i = 0; i < 3; ++i){
			cv::equalizeHist(dst_warp_RGB[i], dst_warp_RGB[i]);

		}
		cv::merge(dst_warp_RGB, 3, his);
		//cv::imshow("After_his", his);
		cv::blur(his, his, cv::Size(3, 3));

		cv::Mat otsu;
		cv::cvtColor(his, otsu, CV_BGR2GRAY);
		/*cv::threshold(otsu, otsu, 0, 255, CV_THRESH_BINARY| CV_THRESH_OTSU);*/
		cv::threshold(otsu, otsu, 0, 255, CV_THRESH_OTSU);
		//cv::adaptiveThreshold(otsu, otsu, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 9, 4);
		//cv::blur(otsu, otsu, cv::Size(3, 3));
		cv::medianBlur(otsu, otsu, 5);
		cv::imshow("After_otsu", otsu);

		//Perform blur
		//blur the image
		////cv::blur(image_clahe, image_clahe, cv::Size(5, 5));
		//cv::medianBlur(image_clahe, image_clahe, 5);

		// display the results  (you might also want to see lab_planes[0] before and after).
		//cv::imshow("image CLAHE", image_clahe);

		//Perform Morphology
		//Create a structuring element (SE)
		int morph_size = 2;
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
		//// Apply the specified morphology operation
		cv::Mat output_grad;
		for (int i = 1; i <50; i++)
		{
			//morphologyEx(image_clahe, output_grad, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), 1);
			morphologyEx(otsu, output_grad, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
			//imshow("output_grad", output_grad);
		}

		/////////////////Morphology end///////////////////////////////

		cv::Mat image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\128.PNG");
		/*cv::Mat image_template_RGB[3];
		cv::split(image_template, image_template_RGB);
		for (int i = 1; i < 2; ++i){
		cv::equalizeHist(image_template_RGB[i], image_template_RGB[i]);

		}
		cv::merge(image_template_RGB, 3, image_template);*/

		cv::Mat out;
		//cv::adaptiveThreshold(dst_warp, dst_warp, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 19);
		//cv::adaptiveThreshold(image_template, image_template, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, 19);
		//medianBlur(dst_warp, dst_warp, 3);
		//medianBlur(image_template, image_template, 3);
		//cv::threshold(dst_warp, dst_warp, 100, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		//cv::blur(dst_warp, dst_warp, cv::Size(3, 3));

		//cv::imshow("after threshold+blur", dst_warp);
		dst_warp = otsu;
		cv::Mat image_matched;
		cv::SiftFeatureDetector detector;
		std::vector<cv::KeyPoint> keypoints1;
		std::vector<cv::KeyPoint> keypoints2;

		detector.detect(dst_warp, keypoints1);
		detector.detect(image_template, keypoints2);

		// Add results to image and save.
		cv::Mat feature_pic1, feature_pic2;
		cv::drawKeypoints(dst_warp, keypoints1, feature_pic1);
		cv::drawKeypoints(image_template, keypoints2, feature_pic2);

		//cv::imshow("sift_result1", feature_pic1);
		//cv::imshow("sift_result2", feature_pic2);

		//计算特征点描述符 / 特征向量提取
		cv::SiftDescriptorExtractor descriptor;
		cv::Mat description1;
		descriptor.compute(dst_warp, keypoints1, description1);
		cv::Mat description2;
		descriptor.compute(image_template, keypoints2, description2);
		//cout << description1.cols << endl;
		//cout << description1.rows << endl;


		//进行BFMatch暴力匹配
		cv::FlannBasedMatcher matcher;
		//cv::BruteForceMatcher<cv::L2<float>>matcher;    //实例化暴力匹配器
		std::vector<cv::DMatch>matches;   //定义匹配结果变量
		matcher.match(description1, description2, matches);  //实现描述符之间的匹配

		int* min;
		//计算向量距离的最大值与最小值：距离越小越匹配
		cv::DMatch maxElement = *std::max_element(matches.begin(), matches.end());
		cv::DMatch minElement = *std::min_element(matches.begin(), matches.end());
		double max_dist = maxElement.distance, min_dist = minElement.distance;
		//匹配结果删选    
		std::vector<cv::DMatch>good_matches;
		for (int i = 0; i<matches.size(); i++)
		{
			if (matches[i].distance < 1 * min_dist)
				good_matches.push_back(matches[i]);
		}

		cv::Mat result;
		drawMatches(dst_warp, keypoints1, image_template, keypoints2, good_matches, result, cv::Scalar(0, 255, 0), cv::Scalar::all(-1));//匹配特征点绿色，单一特征点颜色随机
		for (int i = 0; i<good_matches.size(); i++)
		{
			std::cout << keypoints1[good_matches[i].queryIdx].pt.x << std::endl;
		}
		std::cout << result.cols << ' ' << result.rows << std::endl;
		//45 155 initital
		//width 90
		//height85
		//gap 15
		int k = 0, l = 0;

		for (int i = 0; i < 4; ++i){
			for (int j = 0; j < 4; ++j){
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
				cv::circle(result,
					cv::Point(45 + 90 * (j)+k * 15 * j, 155 + 85 * (i)+l * 15 * i),
					5,
					cv::Scalar(250, 250, 50),
					2,
					8,
					0);

				std::cout << 'x' << 45 + 90 * (j)+k * 15 * j << 'y' << 155 + 85 * (i)+l * 15 * i << std::endl;
			}
		}



		//cv::imshow("a", dst_warp);

		cv::imshow("Match_Result", result);

		for (int i = 0; i < 9; ++i){
			//std::cout << i << std::endl;
			switch (i)
			{
			case 0:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\2.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 1:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\4.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 2:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\8.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 3:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\16.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 4:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\32.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 5:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\64.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 6:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\128.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 7:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\256.PNG", cv::IMREAD_GRAYSCALE);
				break;
			case 8:
				image_template = cv::imread("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\Year3\\Sem1\\数字图像处理\\Software\\qtCyberDIP\\Adaptive\\512.PNG", cv::IMREAD_GRAYSCALE);
				break;
			}

			detector.detect(dst_warp, keypoints1);
			detector.detect(image_template, keypoints2);

			// Add results to image and save.
			cv::drawKeypoints(dst_warp, keypoints1, feature_pic1);
			cv::drawKeypoints(image_template, keypoints2, feature_pic2);

			////cv::imshow("sift_result1", feature_pic1);
			////cv::imshow("sift_result2", feature_pic2);

			////计算特征点描述符 / 特征向量提取
			descriptor.compute(dst_warp, keypoints1, description1);
			descriptor.compute(image_template, keypoints2, description2);
			std::cout << 'c' << dst_warp.cols << std::endl;
			std::cout << 'r' << dst_warp.rows << std::endl;
			std::cout << 'c' << image_template.cols << std::endl;
			std::cout << 'r' << image_template.rows << std::endl;
			std::cout << 'c' << description1.cols << std::endl;
			std::cout << 'r' << description1.rows << std::endl;
			std::cout << 'c' << description2.cols << std::endl;
			std::cout << 'r' << description2.rows << std::endl;
			cv::imshow("des2", description2);
			////进行BFMatch暴力匹配
			matcher.match(description1, description2, matches);  //实现描述符之间的匹配
			std::vector<cv::DMatch>good_matches1;
			////计算向量距离的最大值与最小值：距离越小越匹配
			////匹配结果删选    
			for (int j = 0; j<matches.size(); j++)
			{
				if (matches[j].distance < 3 * min_dist)
					good_matches1.push_back(matches[j]);
			}

			int temp[4][4] = {};
			for (int k = 0; k<good_matches1.size(); k++)
			{
				if (keypoints1[good_matches1[k].queryIdx].pt.x >45 && keypoints1[good_matches1[k].queryIdx].pt.x < 135){
					if (keypoints1[good_matches1[k].queryIdx].pt.y >155 && keypoints1[good_matches1[k].queryIdx].pt.y < 240){
						temp[0][0] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >255 && keypoints1[good_matches1[k].queryIdx].pt.y < 340){
						temp[0][1] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >355 && keypoints1[good_matches1[k].queryIdx].pt.y < 440){
						temp[0][2] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >455 && keypoints1[good_matches1[k].queryIdx].pt.y < 540){
						temp[0][3] += 1;
					}
				}
				else if (keypoints1[good_matches1[k].queryIdx].pt.x >150 && keypoints1[good_matches1[k].queryIdx].pt.x < 240){
					if (keypoints1[good_matches1[k].queryIdx].pt.y >155 && keypoints1[good_matches1[k].queryIdx].pt.y < 240){
						temp[1][0] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >255 && keypoints1[good_matches1[k].queryIdx].pt.y < 340){
						temp[1][1] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >355 && keypoints1[good_matches1[k].queryIdx].pt.y < 440){
						temp[1][2] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >455 && keypoints1[good_matches1[k].queryIdx].pt.y < 540){
						temp[1][3] += 1;
					}
				}
				else if (keypoints1[good_matches1[k].queryIdx].pt.x >255 && keypoints1[good_matches1[k].queryIdx].pt.x < 345){
					if (keypoints1[good_matches1[k].queryIdx].pt.y >155 && keypoints1[good_matches1[k].queryIdx].pt.y < 240){
						temp[2][0] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >255 && keypoints1[good_matches1[k].queryIdx].pt.y < 340){
						temp[2][1] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >355 && keypoints1[good_matches1[k].queryIdx].pt.y < 440){
						temp[2][2] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >455 && keypoints1[good_matches1[k].queryIdx].pt.y < 540){
						temp[2][3] += 1;
					}
				}
				else if (keypoints1[good_matches1[k].queryIdx].pt.x >360 && keypoints1[good_matches1[k].queryIdx].pt.x < 450){
					if (keypoints1[good_matches1[k].queryIdx].pt.y >155 && keypoints1[good_matches1[k].queryIdx].pt.y < 240){
						temp[3][0] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >255 && keypoints1[good_matches1[k].queryIdx].pt.y < 340){
						temp[3][1] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >355 && keypoints1[good_matches1[k].queryIdx].pt.y < 440){
						temp[3][2] += 1;
					}
					else if (keypoints1[good_matches1[k].queryIdx].pt.y >455 && keypoints1[good_matches1[k].queryIdx].pt.y < 540){
						temp[3][3] += 1;
					}
				}

			}
			int cnt = 0;
			int value = 0;
			std::cout << "i: " << pow(2, i + 1) << std::endl;
			int max = temp[0][0];
			for (int k = 0; k < 4; ++k){
				for (int l = 0; l < 4; ++l){
					std::cout << temp[k][l] << ' ' << std::endl;
					//if (temp[k][l] != 0){ //calculate mean of array
					//	++cnt;
					//	value += temp[k][l];	
					//}
					//if (temp[k][l] >max){ max = temp[k][l]; } //get the max value of the matrix

					if (i<2){
						if (temp[k][l] > 2){
							board[k][l] = 1;
							for (int m = 0; m < i + 1; ++m){
								board[k][l] = board[k][l] * 2;
							}
						}
					}
					//else if (i == 6){
					//	if (temp[k][l] > 9){
					//		board[k][l] = 1;
					//		for (int m = 0; m < i + 1; ++m){
					//			board[k][l] = board[k][l] * 2;
					//		}
					//	}
					//}
					else{
						if (temp[k][l] > 19){
							board[k][l] = 1;
							for (int m = 0; m < i + 1; ++m){
								board[k][l] = board[k][l] * 2;
							}
						}
					}

				}
				std::cout << std::endl;
			}
			////////////////////////////Calculate mean by zc/////////////////////////////
			//std::cout << "mean: " << value / cnt << std::endl;
			//for (int k = 0; k < 4; ++k){
			//	for (int l = 0; l < 4; ++l){
			//		if (max - (value / cnt) >= 10){
			//			if (temp[k][l] - (value / cnt) >= 8){//compare current value with mean
			//				board[k][l] = 1;
			//				for (int m = 0; m < i + 1; ++m){
			//					board[k][l] = board[k][l] * 2;
			//				}
			//			}
			//		}
			//		else{
			//			if (temp[k][l] >= (value / cnt)){	//compare current value with mean
			//				board[k][l] = 1;
			//				for (int m = 0; m < i + 1; ++m){
			//					board[k][l] = board[k][l] * 2;
			//				}
			//			}
			//		}

			//		//if (i<3){
			//		//	if (temp[k][l] >= (value/cnt)){	//compare current value with mean
			//		//		board[k][l] = 1;
			//		//		for (int m = 0; m < i + 1; ++m){
			//		//			board[k][l] = board[k][l] * 2;
			//		//		}
			//		//	}
			//		//}
			//		//else{
			//		//	if (temp[k][l]- (value/cnt) >=8){//compare current value with mean
			//		//		board[k][l] = 1;
			//		//		for (int m = 0; m < i + 1; ++m){
			//		//			board[k][l] = board[k][l] * 2;
			//		//		}
			//		//	}
			//		//}
			//	}
			//}
			////std::cout << "cnt: " << cnt << std::endl;
			////std::cout << "value: " << value << std::endl;
			///////////////////////////////////////////////////Calculate mean by zc////////////////////////////////////
			for (int k = 0; k < 4; ++k){
				for (int l = 0; l < 4; ++l){
					std::cout << '[' << board[l][k] << ']';
				}
				std::cout << std::endl;
			}
		}


		///////////////////////////////////////////////////////////template matchin

		//cv::imshow("target", dst_warp);
		//cv::imshow("result", image_matched);
		click = false;
	}
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
