#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencvfunction.h"


using namespace std;
using namespace cv;

int main() {
    string path="H:\\����\\ѧϰ����\\�Ӿ�����\\OpenCV����\\ͼƬ\\�ڰ���Ƶ��ͼƬ\\�����.jpg";
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
