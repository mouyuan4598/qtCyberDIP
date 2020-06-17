cvHOGDescriptor hog(cvSize(32, 32), cvSize(8, 8), cvSize(4, 4), cvSize(4, 4), 10);

	HOG检测器，用来计算HOG描述子的
	int DescriptorDim;HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

	cvMat sampleFeatureMat;所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
	cvMat sampleLabelMat;训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

	CvSVMParams params;
	params.svm_type = CvSVMC_SVC;
	params.kernel_type = CvSVMLINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 3000, 1e-6);
	cvPtrcvSVM svm = CvSVMcreate();SVM分类器

	stdstring ImgName;图片名(绝对路径)

	stdifstream finPos(CUsersmouyuDesktopzcimageimg.txt);正样本图片的文件名列表

	if (!finPos)
	{
		stdcout  PosNeg imglist reading failed...  stdendl;
		return 1;
	}
	for (int num = 0; num  40 && getline(finPos, ImgName); num++)
	{
		stdcout  Now processing original positive image   ImgName  stdendl;
		ImgName = Image + ImgName;加上正样本的路径名
		cvMat src = cvimread(ImgName);读取图片
		cvimshow(src, src);
			if (CENTRAL_CROP)
		cvMat src = in(cvRect(96, 96, 84, 84)); 将96160的INRIA正样本图片剪裁为64128，即剪去上下左右各16个像素
			stdvectorfloat descriptors;HOG描述子向量
			hog.compute(src, descriptors, cvSize(4, 4));计算HOG描述子，检测窗口移动步长(8,8)

			处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = descriptors.size();
				stdcout  DescriptorDim   stdendl; 419832
				初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = cvMatzeros(40, DescriptorDim, CV_32FC1);
				初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = cvMatzeros(40, 1, CV_32SC1);sampleLabelMat的数据类型必须为有符号整数型
			}
			stdcout  descriptors.size()  stdendl; 419832 x 9
			将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i  DescriptorDim; i++){
				stdcout  i  stdendl;
				sampleFeatureMat.atfloat(num, i) = descriptors[i];第num个样本的特征向量中的第i个元素
			}
			stdcout  num  stdendl;
			sampleLabelMat.atint(num, 0) = pow(2, num + 1);正样本类别为1，有人
			if (num % 10 == 0){
				sampleLabelMat.atint(num, 0) = 0;
			}
			else{
				sampleLabelMat.atint(num, 0) = pow(2, (num % 10));
			}

	}
	finPos.close();
	stdcout  Starting training...  stdendl;
	CvSVM SVM;
	SVM.train(sampleFeatureMat, sampleLabelMat, cvMat(), cvMat(), params);
	svm-train(sampleFeatureMat, cvROW_SAMPLE, sampleLabelMat);训练分类器
	stdcout  Finishing training...  stdendl;
	将训练好的SVM模型保存为xml文件
	SVM.save(CUsersmouyuDesktopzcimageSVM_HOG4.xml);
	stdcout  Finishing save file...  stdendl;
	stdcout  SVM.get_support_vector_count()  stdendl; 36
	stdcout  SVM.get_var_count()  stdendl; 419832

