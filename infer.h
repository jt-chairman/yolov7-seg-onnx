#pragma once
#include "global.h"
#include <assert.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>

typedef struct Bbox {
    double x1;
    double y1;
    double x2;
    double y2;
    Bbox(double _x1, double _y1, double _x2, double _y2) {
        x1 = _x1;
        y1 = _y1;
        x2 = _x2;
        y2 = _y2;
    }
    Bbox() {
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 0;
    }
}Bbox;

typedef struct Bboxf
{
    float x1;
    float y1;
    float x2;
    float y2;
    Bboxf(float _x1, float _y1, float _x2, float _y2) {
        x1 = _x1;
        y1 = _y1;
        x2 = _x2;
        y2 = _y2;
    }
    Bboxf() {
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 0;
    }
} Bboxf;

class mycompare

{
public://分别代表重载()和重载后参数列表
    bool operator()(std::pair<double, int> v1, std::pair<double, int> v2)const {
        return v1.first > v2.first;
    }
};