#include "yolov7.h"

using namespace std;
using namespace cv;
using namespace Ort;

inline vector<vector<vector<float>>> acquire_output(vector<Value>& ort_outputs, int num, vector<int64_t> shape) {
	Value* output = &ort_outputs[num];
	float* value = output->GetTensorMutableData<float>();
	vector<vector<vector<float>>> data(shape[0], vector<vector<float>>(shape[1], vector<float>(shape[2], 0)));
	int index = 0;
	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				data[i][j][k] = value[index++];
			}
		}
	}

	return data;
}

inline vector<int64_t> acquire_output_shape(vector<Value>& ort_outputs, int num) {
	Value* output = &ort_outputs[num];
	vector<int64_t> shape = output->GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	return shape;
}

YOLOV7::YOLOV7() {

	 /**
    * 初始化Session选项
    * Available levels are
    * ORT_DISABLE_ALL -> 禁用所有优化
    * ORT_ENABLE_BASIC -> 要启用基本优化(如冗余节点删除)
    * ORT_ENABLE_EXTENDED -> 启用扩展优化(包括1级以上更复杂的优化，如节点融合)
    * ORT_ENABLE_ALL -> 启用所有可能的优化
    **/
	session_options_.SetIntraOpNumThreads(1);
	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); /** 设置图像优化级别 **/

	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

	size_t numInputNodes = session_.GetInputCount();
	size_t numOutputNodes = session_.GetOutputCount();
	AllocatorWithDefaultOptions allocator;

	Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
	auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	auto input_dims = input_tensor_info.GetShape();

	this->inpHeight = input_dims[2];
	this->inpWidth = input_dims[3];

	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();

}

float YOLOV7::iou(Bboxf& box1, Bboxf& box2) {
	double max_x = std::max(box1.x1, box2.x1);  // 找出左上角坐标哪个大
	double max_y = std::max(box1.y1, box2.y1);
	double min_x = std::min(box1.x2, box2.x2);  // 找出右下角坐标哪个小
	double min_y = std::min(box1.y2, box2.y2);
	if (min_x <= max_x || min_y <= max_y) // 如果没有重叠
		return 0;
	float over_area = (min_x - max_x) * (min_y - max_y);  // 计算重叠面积
	float area1 = area(box1);
	float area2 = area(box2);
	float iou = over_area / (area1 + area2 - over_area);
	return iou;
}

float YOLOV7::area(Bboxf& box) {
	float area = (box.x2 - box.x1) * (box.y2 - box.y1);
	return area;
}

vector<vector<vector<float>>> YOLOV7::nonMaxSuppression(vector<vector<vector<float>>> pred, vector<int64_t> shape, int nm){
	// in utils/general.py 
	
	vector<vector<vector<float>>> x_final;
	int bs = shape[0];
	int nc = shape[2] - nm - 5;
	vector<vector<bool>> xc(shape[0], vector<bool>(shape[1], false));
	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			if (pred[i][j][4] > this->confThreshold) {
				xc[i][j] = true;
			}
		}
	}

	int max_wh = 7680;

	int mi = 5 + nc;
	// refer to line #845
	for (int index = 0; index < bs; index++) {
		vector<vector<float>> x;
		vector<vector<float>> tmp = pred[index];
		for (int i = 0; i < xc[index].size(); i++) {
			if (xc[index][i]) {
				x.push_back(tmp[i]);
			}
		}
		// refer to line #861
		for (int i = 0; i < x.size(); i++) {
			for (int j = 5; j < shape[2]; j++) {
				x[i][j] *= x[i][4];
			}
		}

		// refer to line 864
		vector<vector<float>> box = xywh2xyxy(x);
		vector<vector<float>> mask(x.begin(), x.end());

		//refer to line 872
		vector<float> conf;
		vector<int> j;
		for (int i = 0; i < x.size(); i++) {
			vector<float>::iterator loc = max_element(x[i].begin() + 5, x[i].begin() + mi);
			j.push_back(loc - (x[i].begin() + 5));
			conf.push_back(*loc);
		}
		vector<vector<float>> x_prime;
		for (int i = 0; i < x.size(); i++) {
			if (conf[i] < confThreshold) {
				continue;
			}
			vector<float> tmp;
			for (int ii = 0; ii < 4; ii++) {
				tmp.push_back(box[i][ii]);
			}
			tmp.push_back(conf[i]);
			tmp.push_back((float)j[i]);
			for (int ii = 7; ii < shape[2]; ii++) {
				tmp.push_back(mask[i][ii]);
			}
			x_prime.push_back(tmp);
		}

		//refer to line 883
		int n = x_prime.size();
		if (!n) {
			continue;
		}
		else {
			sort(x_prime.begin(), x_prime.end(), [](vector<float> a, vector<float> b) {return a[4] > b[4]; });
		}

		//refer to line 893
		vector<float> scores;
		vector<vector<float>> boxes;
		for (int i = 0; i < x_prime.size(); i++) {
			vector<float> tmp;
			scores.push_back(x_prime[i][4]);
			float c = x_prime[i][5] * max_wh;
			for (int ii = 0; ii < 4; ii++) {
				tmp.push_back(x_prime[i][ii] + c);
			}
			boxes.push_back(tmp);
		}

		for (int i = 0; i < boxes.size() - 1; i++) {
			if (excluded_indices.find(i) != excluded_indices.end()) {
				continue;
			}
			for (int j = i + 1; j < boxes.size(); j++) {
				Bboxf a(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
				Bboxf b(boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]);
				float iou_value = iou(a, b);
				if (iou_value > nms_threshold) {
					excluded_indices.insert(j);
				}
			}
		}

		//refer to line 906
		vector<vector<float>> x_nms;
		for (int i = 0; i < x_prime.size(); i++) {
			if (excluded_indices.find(i) == excluded_indices.end()) {
				x_nms.push_back(x_prime[i]);
			}
		}
		x_final.emplace_back(x_nms);
	}
	return x_final;
}

vector<vector<float>> YOLOV7::xywh2xyxy(vector<vector<float>> v_xywh) {
	vector<vector<float>> bbox;
	for (int i = 0; i < v_xywh.size();i++) {
		vector<float> tmp;
		tmp.push_back(v_xywh[i][0] - v_xywh[i][2] / 2);
		tmp.push_back(v_xywh[i][1] - v_xywh[i][3] / 2);
		tmp.push_back(v_xywh[i][0] + v_xywh[i][2] / 2);
		tmp.push_back(v_xywh[i][1] + v_xywh[i][3] / 2);
		bbox.push_back(tmp);
	}
	return bbox;
}

void YOLOV7::crop(vector<Mat>& masks, vector<vector<float>> boxes, int h, int w) {
	for (int k = 0; k < masks.size(); k++) {
		int left = boxes[k][0];
		int right = boxes[k][2];
		int top = boxes[k][1];
		int bottom = boxes[k][3];
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (j < left || j > right || i < top || i > bottom) {;
					masks[k].at<uchar>(i, j) = 0;
				}
			}
		}
	}

}

void YOLOV7::process_mask(vector<vector<vector<float>>> proto, vector<int64_t> proto_shape, vector<vector<float>> det, int ih, int iw) {
	// in segment/general.py
	
	vector<vector<float>> masks_in;
	vector<vector<float>> bboxes;
	for (int i = 0; i < det.size(); i++) {
		vector<float> m_tmp;
		vector<float> b_tmp;
		for (int ii = 6; ii < det[i].size(); ii++) {
			m_tmp.push_back(det[i][ii]);
		}
		for (int ii = 0; ii < 4; ii++) {
			b_tmp.push_back(det[i][ii]);
		}
		masks_in.push_back(m_tmp);
		bboxes.push_back(b_tmp);
	}
	int c = proto_shape[0], mh = proto_shape[1], mw = proto_shape[2];

	// refer to line 55
	Mat masks_in_m(masks_in.size(), masks_in[0].size(), CV_32F);
	for (int i = 0; i < masks_in.size(); i++) {
		for (int j = 0; j < masks_in[0].size(); j++) {
			masks_in_m.at<float>(i, j) = masks_in[i][j];
		}
	}

	Mat proto_2d(c, mh * mw, CV_32F);
	for (int i = 0; i < c; i++) {
		for (int j = 0; j < mh * mw; j++) {
			proto_2d.at<float>(i, j) = proto[i][j / mw][j % mw];
		}
	}

	Mat masks_2d = masks_in_m * proto_2d;
	cv::exp(-masks_2d, masks_2d);
	masks_2d = 1 / (1 + masks_2d);


	for (vector<float>& box : bboxes) {
		box[0] *= (float)mw / iw;
		box[2] *= (float)mw / iw;
		box[1] *= (float)mh / ih;
		box[3] *= (float)mh / ih;
	}
	float* p = (float*)masks_2d.data;
	for (int i = 0; i < bboxes.size(); i++) {
		Mat tmp(Size(mw, mh), CV_8U);
		for (int j = 0; j < mh * mw; j++) {
			tmp.at<uchar>(j / mw, j % mw) = (int)(p[i * mh * mw + j] * 255);
		}
		m_masks.push_back(tmp);
	}


	crop(m_masks, bboxes, mh, mw);
	for (int i = 0; i < m_masks.size(); i++) {
		resize(m_masks[i], m_masks[i], cv::Size(iw, ih));
		threshold(m_masks[i], m_masks[i], 126, 255, THRESH_BINARY);
	}
}

void YOLOV7::scale_coords(vector<vector<float>>& boxes, int src_w, int src_h, int w, int h){
	// in utils/general.py
	// refer to line 777
	float gain = std::min((float)w / src_w, (float)h / src_h);
	float pad[2](((float)w - src_w * gain) / 2, ((float)h - src_h * gain) / 2);
	for (vector<float>& box : boxes) {
		for (int i = 0; i < 4; i++) {
			if (i == 0 || i == 2) {
				box[i] -= pad[0];
			}
			else {
				box[i] -= pad[1];
			}
			box[i] /= gain;
		}
	}
	// refer to function clip_coordinate in line 793
	for (vector<float>& box : boxes) {
		for (int i = 0; i < 4; i++) {
			box[i] = box[i] > 0 ? box[i] : 0;
			if (i == 0 || i == 2) {
				box[i] = box[i] < src_w ? box[i] : src_w;
			}
			else{
				box[i] = box[i] < src_h ? box[i] : src_h;
			}
		}
	}
	for (int i = 0; i < boxes.size(); i++) {
		vector<int> tmp;
		for (int j = 0; j < 4; j++) {
			tmp.push_back((int)boxes[i][j]);
		}
		m_bboxes.push_back(tmp);
	}
}

vector<float> YOLOV7::preprocess(cv::Mat& img) {
	float img_w = img.cols, img_h = img.rows;
	float scale = min(inpHeight / img_h, inpWidth / img_w);
	int new_w = int(img_w * scale + 0.5);
	int new_h = int(img_h * scale + 0.5);
	resize(img, img, Size(0, 0), scale, scale);

	Mat new_img = Mat::ones(inpHeight, inpWidth, img.type()) * 114;
	img.copyTo(new_img(Range((inpHeight - new_h) / 2, (inpHeight + new_h) / 2), Range((inpWidth - new_w) / 2, (inpWidth + new_w) / 2)));
	img = new_img.clone();

	cvtColor(img, img, cv::COLOR_GRAY2RGB);
	img.convertTo(img, CV_32F);
	vector<float> input_data(this->inpWidth * this->inpHeight * 3);
	int counter = 0;
	for (unsigned k = 0; k < 3; k++)
	{
		for (unsigned i = 0; i < img.rows; i++)
		{
			for (unsigned j = 0; j < img.cols; j++)
			{
				input_data[counter++] = (img.at<cv::Vec3f>(i, j)[k]) / 255;
			}
		}
	}
	return input_data;
}

void YOLOV7::run(Mat frame){
	Mat rect = frame.clone();
	cvtColor(rect, rect, COLOR_GRAY2RGB);
	int ih = frame.rows;
	int iw = frame.cols;
	vector<float> input_data = preprocess(frame);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_data.data(), input_data.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = session_.Run(RunOptions{ nullptr }, &input_name_vec[0], &input_tensor_, 1, output_name_vec.data(), output_name_vec.size());   // 开始推理

	vector<int64_t> shape0 = acquire_output_shape(ort_outputs, 0);
	vector<vector<vector<float>>> pred = acquire_output(ort_outputs, 0, shape0);
	vector<int64_t> shape4 = acquire_output_shape(ort_outputs, 4);
	shape4.erase(shape4.begin());
	vector<vector<vector<float>>> proto = acquire_output(ort_outputs, 4, shape4);

	vector<vector<vector<float>>> det = nonMaxSuppression(pred, shape0);

	for (int i = 0; i < det.size(); i++) {
		process_mask(proto, shape4, det[i], inpHeight, inpWidth);
		scale_coords(det[i], iw, ih, inpWidth, inpHeight);
	}

	for (int i = 0; i < det[0].size(); i++) {
		imwrite(save_path + "mask#" + to_string(i) + ".bmp", m_masks[i]);
		cout << save_path + "mask#" + to_string(i) + ".bmp" << endl;
		rectangle(rect, cv::Point((int)det[0][i][0], (int)det[0][i][1]), cv::Point((int)det[0][i][2], (int)det[0][i][3]), cv::Scalar(0, 255, 0));
	}
	imwrite(save_path + "rect" + ".bmp", rect);

}

