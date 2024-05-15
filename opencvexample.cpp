//
// Created by HS-YF-1 on 2024/5/15.
//
#include "opencvfunction.h"
using namespace cv;

void QuickDemo::image_colorization(String imageFileName) {
    Mat image= imread(imageFileName);
    imshow("≤‚ ‘",image);

}