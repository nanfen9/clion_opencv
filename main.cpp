#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencvfunction.h"


using namespace std;
using namespace cv;

int main() {
    string path="H:\\资料\\学习资料\\视觉资料\\OpenCV资料\\图片\\黑白视频和图片\\刘亦菲.jpg";
    Mat image= imread(path);
    QuickDemo qd;
    //qd.face_detection_demo();
    qd.image_colorization(path);
    qd.grayscale_is_colored(image);
    cout << "Hello, World!" << std::endl;
    waitKey(0);
    destroyAllWindows();
    return 0;
}
