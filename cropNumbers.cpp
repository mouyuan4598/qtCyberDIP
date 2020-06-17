cv::Mat img = cv::imread("C:\\Users\\mouyu\\Desktop\\DIP\\warp board.PNG");
	std::cout << img.rows << ' ' << img.cols;
	int k = 0, l = 0;
	cv::Mat crop[4][4];
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
		/*	cv::circle(img,
				cv::Point(40 + 85 * (j)+k * 15 * j, 150 + 75 * (i)+l * 15 * i),
				5,
				cv::Scalar(250, 250, 50),
				2,
				8,
				0);*/

			//std::cout << 'x' << 45 + 90 * (j)+k * 15 * j << 'y' << 155 + 85 * (i)+l * 15 * i << std::endl;
			crop[i][j] = img(cv::Rect(35 + 96 * (j)+ k * 5 * j, 145 + 84 * (i)+ l * 5 * i, 96, 84));
		}
		
	}
	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 4; ++j){
			cv::imshow("c" + std::to_string(i) + std::to_string(j) + ".PNG", crop[i][j]);
			cv::imwrite("C:\\Users\\mouyu\\Desktop\\DIP\\c" + std::to_string(i) + std::to_string(j) + ".PNG", crop[i][j]);
		}
	}
	