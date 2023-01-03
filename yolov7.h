#pragma once

#include "infer.h"
#include <cmath>
#include <vector>

class YOLOV7{
public:
	YOLOV7();
	void run(cv::Mat frame);
	std::vector<std::vector<std::vector<float>>> nonMaxSuppression(std::vector< std::vector<std::vector<float>>> pred, std::vector<int64_t> shape, int nm = 32);
	void process_mask(std::vector<std::vector<std::vector<float>>> proto, std::vector<int64_t> proto_shape, std::vector<std::vector<float>> det, int ih, int iw);
	std::vector<std::vector<float>> xywh2xyxy(std::vector<std::vector<float>> v_xywh);
	void crop(std::vector<cv::Mat>& masks, std::vector<std::vector<float>> boxes, int w, int h);
	void scale_coords(std::vector<std::vector<float>>& boxes, int src_w, int src_h, int w, int h);
	std::vector<float> preprocess(cv::Mat& img);


	float area(Bboxf& box);
	float iou(Bboxf& box1, Bboxf& box2);

	std::vector<cv::Mat> m_masks;
	std::vector<std::vector<int>> m_bboxes;
private:
	int inpWidth = 0;
	int inpHeight = 0;
	int num_proposal = 0;
	std::vector<std::string> class_names = { "front", "back" };
	int num_class = class_names.size();

	double confThreshold = 0.25;
	double nms_threshold = 0.45;
	std::set<int> excluded_indices;
	std::string save_path = "../results/";


	std::vector<char*> input_name_vec = {(char*)"images"};
	std::vector<char*> output_name_vec = { (char*)"output",(char*)"onnx::Slice_531",(char*)"onnx::Slice_638" ,(char*)"onnx::Slice_744",(char*)"516" };
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_FATAL, "YOLOV7");
	Ort::SessionOptions session_options_ = Ort::SessionOptions();
	//*************************************************************************
// 创建Session并把模型载入内存
	Ort::Session session_ = Ort::Session(env, (const wchar_t*)L"../models/best.onnx", session_options_);
};