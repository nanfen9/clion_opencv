//
// Created by HS-YF-1 on 2024/5/15.
//
#include <opencv2/dnn.hpp>
#include "opencvfunction.h"

using namespace std;
using namespace cv::dnn;

//图像色彩空间转换
void QuickDemo::colorSpace_Demo(Mat&image) {
    Mat gray, hsv;
    //色彩空间转换函数 COLOR_BGR2GRAY=6 彩色转灰度 COLOR_GRAY2BGR=8 灰度转彩色
    //                 COLOR_BGR2HSV=40 BGR转HSV   COLOR_HSV2BGR=54 HSV转BGR
    cvtColor(image, hsv, COLOR_BGR2HSV);
    //H 0~180， S,V都是255
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("HSV", hsv);
    imshow("灰度", gray);
    //保存图像
    //imwrite("D:/hsv.bmp", hsv);
    //imwrite("D:/gray.png", gray);
}
//图像对象的创建与赋值
void QuickDemo::mat_creation_demo(Mat& image) {
    Mat m1, m2;
    //克隆
    //m1 = image.clone();
    //复制
    //image.copyTo(m2);
    //创建空白图像
    Mat m3 = Mat::zeros(Size(400, 400), CV_8UC3);
    Mat m4 = Mat::ones(Size(8, 8), CV_8UC1);
    //给图像赋值颜色
    m3 =Scalar(255,0,0);
    //显示信息
    //std::cout << "Width:" << m3.cols << "heigth::" << m3.rows << "channels:" << m3.channels() << std::endl;
    //std::cout << m3 << std::endl;
    imshow("创建图像", m3);
    //赋值法
    Mat m5 = m3;
    m5 = Scalar(0,255,255);
    imshow("图像", m3);
    //直接赋值创建图像
    Mat kernel = (Mat_ <char>(3, 3) << 0, -1, 0,
            -1, 5, -1,
            0, -1, 0
    );
    imshow("创建图像1", kernel);
}
//图像像素的读写操作
void QuickDemo::pixel_visit_demo(Mat&image) {
    int w = image.cols;
    int h = image.rows;
    int dims = image.channels();
    //for (int row=0;row<h;row++)
    //{
    //	for (int col=0;col<w;col++)
    //	{
    //		if (dims==1)//灰度图像
    //		{
    //			int pv = image.at<uchar>(row, col);
    //			image.at<uchar>(row, col) = 255 - pv;
    //		}
    //		if (dims==3)//彩色图像
    //		{
    //			Vec3b bgr= image.at<Vec3b>(row, col);
    //			image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
    //			image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
    //			image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
    //		}
    //	}
    //}
    for (int row = 0; row < h; row++)
    {
        uchar* current_row = image.ptr<uchar>(row);
        for (int col = 0; col < w; col++)
        {
            if (dims == 1)//灰度图像
            {
                int pv = *current_row;
                *current_row++ = 255 - pv;
            }
            if (dims == 3)//彩色图像
            {
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
            }
        }
    }
    imshow("像素读写演示", image);
}
//图像像素的算术操作
void QuickDemo::operators_demo(Mat&image) {
    Mat dst= Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    m = Scalar(5, 5, 5);
    //乘法函数
    //multiply(image, m, dst);
    //加法操作
    //dst = image + Scalar(50,50,50);

    //加法
    /*
    int w = image.cols;
    int h = image.rows;
    int dims = image.channels();
    for (int row=0;row<h;row++)
    {
        for (int col=0;col<w;col++)
        {
            Vec3b p1 = image.at<Vec3b>(row, col);
            Vec3b p2 = m.at<Vec3b>(row, col);
            dst.at<Vec3b>(row, col)[0] =saturate_cast<uchar>( p1[0] - p2[0]);
            dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] - p2[1]);
            dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] - p2[2]);
        }
    }
    */
    //加法
    //add(image, m, dst);
    //减法
    //subtract(image, m, dst);
    //除法
    divide(image, m, dst);

    imshow("算术操作", dst);
}
//
static void on_track(int b, void* userdata) {
    Mat image = *((Mat*)userdata);
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    //m = Scalar(b, b, b);
    //add(image, m, dst);
    addWeighted(image, 1.0, m, 0, b, dst);
    imshow("亮度与对比度调整", dst);
}
static void on_contrast(int b, void* userdata) {
    Mat image = *((Mat*)userdata);
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    double contrast = b / 100.0;
    addWeighted(image, contrast, m, 0.0, 0, dst);
    imshow("亮度与对比度调整", dst);
}
//滚动条操作-调整图像亮度和对比度
void QuickDemo::tracking_bar_demo(Mat&image) {
    namedWindow("亮度与对比度调整", WINDOW_AUTOSIZE);
    int lightness = 50;
    int max_value = 100;
    int contrast_value = 100;
    //滚动条
    createTrackbar("Value Bar:", "亮度与对比度调整", &lightness, max_value, on_track,(void*)(&image));
    createTrackbar("Contrast Bar:", "亮度与对比度调整", &contrast_value, 200, on_contrast, (void*)(&image));
    on_track(50, &image);
}
//键盘事件
void QuickDemo::key_demo(Mat&image) {
    Mat dst=Mat::zeros(image.size(),image.type());
    while (true)
    {
        //每次等待100毫秒--做视频时是1
        int c = waitKey(100);
        if (c == 27) {
            break;
        }
        if (c==49)//Key#1
        {
            std::cout << "you enter key #1" << std::endl;
            cvtColor(image, dst, COLOR_BGR2GRAY);
            //imshow("键盘响应", dst);
        }
        if (c == 50)//Key#2
        {
            std::cout << "you enter key #2" << std::endl;
            cvtColor(image, dst, COLOR_BGR2HSV);
            //imshow("键盘响应", dst);
        }
        if (c == 51)//Key#3
        {
            std::cout << "you enter key #3" << std::endl;
            dst = Scalar(50, 50, 50);
            add(image, dst, dst);
        }
        imshow("键盘响应", dst);
    }
}
//颜色表操作函数
void QuickDemo::color_style_demo(Mat&image) {
    int colormap[] = {
            COLORMAP_AUTUMN,
            COLORMAP_BONE,
            COLORMAP_JET,
            COLORMAP_WINTER,
            COLORMAP_RAINBOW,
            COLORMAP_OCEAN,
            COLORMAP_SUMMER,
            COLORMAP_SPRING,
            COLORMAP_COOL,
            COLORMAP_HSV,
            COLORMAP_PINK,
            COLORMAP_HOT,
            COLORMAP_PARULA,
            COLORMAP_MAGMA,
            COLORMAP_INFERNO,
            COLORMAP_PLASMA,
            COLORMAP_VIRIDIS,
            COLORMAP_CIVIDIS,
            COLORMAP_TWILIGHT,
            COLORMAP_TWILIGHT_SHIFTED,
            COLORMAP_TURBO,
            COLORMAP_DEEPGREEN,


    };
    Mat dst;
    int index = 0;
    while (true)
    {
        //每次等待100毫秒--做视频时是1
        int c = waitKey(2000);
        if (c == 27) {
            break;
        }
        //颜色表操作函数
        applyColorMap(image, dst, colormap[index % 21]);
        index++;
        imshow("颜色变换", dst);
    }
}
//图像像素的逻辑操作
void QuickDemo::bitwise_demo(Mat&image) {
    //创建空白图像
    Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
    Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
    //绘制矩形
    rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
    rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
    imshow("m1", m1);
    imshow("m2", m2);
    Mat dst;
    //与操作
    //bitwise_and(m1, m2, dst);
    //或操作
    //bitwise_or(m1, m2, dst);
    //取反操作-非
    //bitwise_not(image, dst);
    //dst = ~image;
    //异或操作
    bitwise_xor(m1, m2, dst);
    imshow("像素位操作", dst);
}
//图像通道分离与合并
void QuickDemo::channels_demo(Mat&image) {
    //容器
    std::vector<Mat>mv;
    //分离
    split(image, mv);
    imshow("蓝色", mv[0]);
    imshow("绿色", mv[1]);
    imshow("红色", mv[2]);

    Mat dst;
    mv[1] = 0;
    mv[2] = 0;
    //合并
    merge(mv, dst);
    imshow("蓝色", dst);
    //通道混合
    int from_to[] = { 0,2,1,1,2,0 };
    mixChannels(&image,1, &dst,1, from_to,3);
    imshow("通道混合", dst);
}
//图像色彩空间转换----进一步学习
void QuickDemo::inrange_demo(Mat&image) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    Mat mask;

    inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
    imshow("mask", mask);
    Mat redback = Mat::zeros(image.size(), image.type());
    redback = Scalar(40, 40, 200);
    bitwise_not(mask, mask);
    imshow("mask", mask);
    //图像拷贝
    image.copyTo(redback, mask);
    imshow("ROI区域提取", mask);
}
//图像像素值统计
void QuickDemo::pixel_statistic_demo(Mat&image) {
    double minv, maxv;
    Point minloc, maxloc;
    //图像最小值最大值函数
    std::vector<Mat>mv;
    //分离
    split(image, mv);
    for (int i=0;i<mv.size();i++)
    {
        minMaxLoc(mv[i], &minv, &maxv, &minloc, &maxloc, Mat());
        std::cout <<"No.channels"<<i<< "min value" << minv << "max value" << maxv << std::endl;
    }

    //图像均值和方差
    Mat mean, stddev;
    Mat redback = Mat::zeros(image.size(), image.type());
    redback = Scalar(40, 40, 200);
    imshow("redback", redback);
    meanStdDev(image, mean, stddev);
    mean.at<double>(1, 0);
    std::cout << "means:" << mean << "stddev:" << stddev << std::endl;
}
//图像几何形状创建
void QuickDemo::drawing_demo(Mat&image) {
    Rect rect;
    rect.x = 100;
    rect.y = 100;
    rect.width = 100;
    rect.height = 100;
    Mat bg = Mat::zeros(image.size(), image.type());
    //绘制矩形
    rectangle(bg, rect, Scalar(0, 0, 255), 3, 8, 0);
    //绘制圆
    circle(bg, Point(150, 200), 50, Scalar(255, 0, 0), 3, 8, 0);
    //绘制线段
    line(bg, Point(100, 100), Point(200, 200), Scalar(0, 255, 0), 3, 8, 0);
    //绘制椭圆
    RotatedRect rrt;
    rrt.center = Point(100, 100);
    rrt.size = Size(100, 150);
    rrt.angle = 0.0;
    ellipse(bg, rrt, Scalar(50, 255, 50), 3, 8);
    Mat dst;
    addWeighted(image, 0.7, bg, 0.3, 0, dst);
    imshow("绘制演示", dst);
}
//随机数和随机颜色
void QuickDemo::random_drawing() {
    Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
    int w = canvas.cols;
    int h = canvas.rows;
    RNG rng(12345);
    while (true)
    {
        //每次等待100毫秒--做视频时是1
        int c = waitKey(10);
        if (c == 27) {
            break;
        }
        int x1 = rng.uniform(0, w);
        int y1 = rng.uniform(0, h);
        int x2 = rng.uniform(0, w);
        int y2 = rng.uniform(0, h);
        int b = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int r = rng.uniform(0, 255);
        //canvas = Scalar(0, 0, 0);
        line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b,g,r), 1, LINE_AA, 0);
        imshow("随机绘制", canvas);
    }
}
//多边形填充和绘制
void QuickDemo::polyline_drawing_demo() {
    Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
    Point p1(100, 100);
    Point p2(120, 120);
    Point p3(140, 160);
    Point p4(180, 200);
    Point p5(60, 220);
    std::vector<Point>pts;
    pts.push_back(p1);
    pts.push_back(p2);
    pts.push_back(p3);
    pts.push_back(p4);
    pts.push_back(p5);
    //绘制多边形区域
    fillPoly(canvas, pts, Scalar(255, 255, 0), 8, 0);
    //绘制多边形轮廓
    polylines(canvas, pts, true, Scalar(0, 0, 255), 2, LINE_AA, 0);
    //绘制多个多边形
    std::vector<std::vector<Point>>contours;
    contours.push_back(pts);
    drawContours(canvas, contours, -1, Scalar(0, 255, 0), 2);
    imshow("多边形绘制", canvas);
}
//鼠标操作与响应
Point sp(-1, -1);
Point ep(-1, -1);
Mat temp;
static void on_draw(int event, int x, int y, int flags, void *userdata) {
    Mat image = *((Mat*)userdata);
    if (event == EVENT_LBUTTONDOWN) {
        sp.x = x;
        sp.y = y;
        std::cout << "start point:" << sp << std::endl;
    }
    else if (event == EVENT_LBUTTONUP) {
        ep.x = x;
        ep.y = y;
        int dx = ep.x - sp.x;
        int dy = ep.y - sp.y;
        if (dx > 0 && dy > 0) {
            Rect box(sp.x, sp.y, dx, dy);
            temp.copyTo(image);
            imshow("ROI区域", image(box));
            rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
            imshow("鼠标绘制", image);
            // ready for next drawing
            sp.x = -1;
            sp.y = -1;
        }
    }
    else if (event == EVENT_MOUSEMOVE) {
        if (sp.x > 0 && sp.y > 0) {
            ep.x = x;
            ep.y = y;
            int dx = ep.x - sp.x;
            int dy = ep.y - sp.y;
            if (dx > 0 && dy > 0) {
                Rect box(sp.x, sp.y, dx, dy);
                temp.copyTo(image);
                rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
                imshow("鼠标绘制", image);
            }
        }
    }
}
void QuickDemo::mousse_drawing_demo(Mat&image) {
    namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
    setMouseCallback("鼠标绘制", on_draw,(void*)(&image));
    imshow("鼠标绘制", image);
    temp = image.clone();
}
//图像像素类型转换与归一化
void QuickDemo::norm_demo(Mat&image) {
    Mat dst;
    std::cout << image.type() << std::endl;
    //图像格式变换--32位
    image.convertTo(image, CV_32F);//CV_32S
    std::cout << image.type() << std::endl;
    normalize(image, dst, 1.0, 0,NORM_MINMAX);
    std::cout << dst.type() << std::endl;
    imshow("图像归一化", dst);
}
//图像缩放与插值
void QuickDemo::resize_demo(Mat&image) {
    Mat zoomin, zoomout;
    int h = image.rows;
    int w = image.cols;
    resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
    imshow("zoomin", zoomin);
    resize(image, zoomout, Size(w * 1.5, h * 1.5), 0, 0, INTER_LINEAR);
    imshow("zoomout", zoomout);
}
//图像翻转
void QuickDemo::flip_demo(Mat&image) {
    Mat dst;
    flip(image, dst, 0);//0--上下翻转
    flip(image, dst, 1);//1--左右翻转
    flip(image, dst, -1);//0--180°旋转
    imshow("图像翻转", dst);
}
//图像旋转
void QuickDemo::rotate_demo(Mat&image) {
    Mat dst, M;
    int w = image.cols;
    int h = image.rows;
    //生成矩阵
    M = getRotationMatrix2D(Point(w / 2, h / 2), 85, 1.0);
    //获取旋转的cos和sin
    double cos = abs(M.at<double>(0, 0));
    double sin = abs(M.at<double>(0, 1));
    //旋转后的宽和高
    int nw = cos * w + sin * h;
    int nh = sin * w + cos * h;
    //旋转后的中心
    M.at<double>(0, 2) = M.at<double>(0, 2) + (nw / 2 - w / 2);
    M.at<double>(1, 2) = M.at<double>(1, 2) + (nh / 2 - h / 2);
    //图像旋转
    warpAffine(image, dst, M, Size(nw,nh),INTER_LINEAR,0,Scalar(255,0,0));
    imshow("旋转演示", dst);
}
//视频文件摄像头使用和视频处理与保存
void QuickDemo::video_demo(Mat&image) {
    //读取视频文件和摄像头
    VideoCapture capture("D:/软件/测试图片/城市.mp4");
    //获取视频的参数
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(CAP_PROP_FRAME_COUNT);
    double fps = capture.get(CAP_PROP_FPS);
    //显示
    std::cout << "frame_width:" << frame_width << std::endl;
    std::cout << "frame_height:" << frame_height << std::endl;
    std::cout << "Number of Frames:" << count << std::endl;
    std::cout << "FPS:" << fps << std::endl;
    //声明保存的视频路径和格式
    VideoWriter writer("D:/软件/测试图片/2.mp4",capture.get(CAP_PROP_FOURCC),fps,Size(frame_width, frame_height),true);
    Mat frame;
    while (true)
    {
        capture.read(frame);
        flip(frame, frame, 1);
        if (frame.empty())
        {
            break;
        }
        imshow("frame", frame);
        colorSpace_Demo(frame);
        //保存视频
        writer.write(frame);
        int c = waitKey(10);
        if (c == 27) {
            break;
        }
    }
    //清楚视频内存
    capture.release();
    writer.release();
}
//图像直方图
void QuickDemo::histogram_demo(Mat&image) {
    //三通道分离
    std::vector<Mat>bgr_plane;
    split(image, bgr_plane);
    //定义参数变量
    const int channels[1] = { 0 };
    const int bins[1] = { 256 };
    float hranges[2] = { 0,255 };
    const float* ranges[1] = { hranges };
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    //计算Blue，green，red通道的直方图
    calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
    //显示直方图
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / bins[0]);
    Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
    //归一化直方图数据
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    //绘制直方图曲线
    for (int i=1;i<bins[0];i++)
    {
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
             Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
             Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
             Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
    }
    namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
    imshow("Histogram Demo", histImage);
}
//二维直方图
void QuickDemo::histogram_2d_demo(Mat&image) {
    //2D直方图
    Mat hsv, hs_hist;
    //转换成hsv通道图像
    cvtColor(image, hsv, COLOR_BGR2HSV);
    //定义参数
    int hbins = 30, sbins = 32;
    int hist_bins[] = { hbins,sbins };
    float h_range[] = { 0,180 };
    float s_range[] = { 0,256 };
    const float* hs_ranges[] = { h_range,s_range };
    int hs_channels[] = { 0,1 };
    //获取图像的二维直方图
    calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
    double maxVal = 0;
    minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
    int scale = 10;
    Mat hist2d_image = Mat::zeros(sbins*scale, hbins*scale, CV_8UC3);
    //绘制二维直方图
    for (int h=0;h<hbins;h++)
    {
        for (int s=0;s<sbins;s++)
        {
            float binVal = hs_hist.at<float>(h, s);
            int intensity = cvRound(binVal * 255 / maxVal);
            rectangle(hist2d_image, Point(h*scale, s*scale),
                      Point((h + 1)*scale - 1, (s + 1)*scale - 1),
                      Scalar::all(intensity), -1);
        }
    }
    applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);
    imshow("H-S Histogram", hist2d_image);

}
//直方图均衡化
void QuickDemo::histogram_eq_demo(Mat&image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat dst;
    //直方图均衡化
    equalizeHist(gray, dst);
    imshow("直方图均衡化演示", dst);
}
//图像卷积操作
void QuickDemo::blur_demo(Mat&image) {
    Mat dst;
    blur(image, dst, Size(3, 3), Point(-1, -1));
    imshow("图像模糊", dst);
}
//高斯模糊
void QuickDemo::gaussian_blur_demo(Mat&image) {
    Mat dst;
    GaussianBlur(image, dst, Size(5, 5), 15);
    imshow("高斯模糊", dst);
}
//高斯双边模糊
void QuickDemo::bifilter_demo(Mat&image) {
    Mat dst;
    bilateralFilter(image, dst, 0, 100, 10);
    imshow("双边模糊", dst);
}
//实时人脸检测
void QuickDemo::face_detection_demo(){
    //定义模型路径
    string root_dir = "E:/OpenCV/OpenCV4.8/opencv/sources/samples/dnn/face_detector/";
    //导入模型路径
    Net net=readNetFromTensorflow(root_dir + "opencv_face_detector_uint8.pb", root_dir + "opencv_face_detector.pbtxt");
    //读取视频或图像文件
    VideoCapture capture(0);
    Mat frame;
    while (true)
    {
        //将读取的文件存到变量
        capture.read(frame);

        flip(frame, frame, 1);
        if (frame.empty())
        {
            break;
        }
        //加载模型参数
        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
        //导入模型
        net.setInput(blob);//NCHW
        //模型推理
        Mat probs = net.forward();//
        Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
        //解析结果
        for (int i=0;i<detectionMat.rows;i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence>0.5)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3)*frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4)*frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5)*frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6)*frame.rows);
                Rect box(x1, y1, x2 - x1, y2 - y1);
                rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
        imshow("人脸检测演示", frame);
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
}
//图像铅笔化和卡通化
void QuickDemo::image_cartoonization_demo(Mat& image)
{
    Mat imgGray, imgColor;
    //灰色铅笔画、彩色铅笔画效果
    pencilSketch(image, imgGray, imgColor, 5, 0.1f, 0.03f);
    Mat result;
    //卡通化效果
    stylization(image, result, 5, 0.6);
    imshow("灰色铅笔化效果", imgGray);
    imshow("彩色铅笔化效果", imgColor);
    imshow("卡通画效果", result);
}
// the 313 ab cluster centers from pts_in_hull.npy (already transposed)
static float hull_pts[] = {
        -90., -90., -90., -90., -90., -80., -80., -80., -80., -80., -80., -80., -80., -70., -70., -70., -70., -70., -70., -70., -70.,
        -70., -70., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -50., -50., -50., -50., -50., -50., -50., -50.,
        -50., -50., -50., -50., -50., -50., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -30.,
        -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -20., -20., -20., -20., -20., -20., -20.,
        -20., -20., -20., -20., -20., -20., -20., -20., -20., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.,
        -10., -10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
        20., 20., 20., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 40., 40., 40., 40.,
        40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
        50., 50., 50., 50., 50., 50., 50., 50., 50., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
        60., 60., 60., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 80., 80., 80.,
        80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 90., 90., 90., 90., 90., 90., 90., 90., 90., 90.,
        90., 90., 90., 90., 90., 90., 90., 90., 90., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 50., 60., 70., 80., 90.,
        20., 30., 40., 50., 60., 70., 80., 90., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -20., -10., 0., 10., 20., 30., 40., 50.,
        60., 70., 80., 90., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -40., -30., -20., -10., 0., 10., 20.,
        30., 40., 50., 60., 70., 80., 90., 100., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -50.,
        -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -60., -50., -40., -30., -20., -10., 0., 10., 20.,
        30., 40., 50., 60., 70., 80., 90., 100., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
        100., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -80., -70., -60., -50.,
        -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -90., -80., -70., -60., -50., -40., -30., -20., -10.,
        0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30.,
        40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70.,
        80., -110., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100.,
        -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100., -90., -80., -70.,
        -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -110., -100., -90., -80., -70., -60., -50., -40., -30.,
        -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0.
};
//灰度图彩色化
void QuickDemo::grayscale_is_colored(Mat& image)
{

    string modelTxt = "H:\\资料\\学习资料\\视觉资料\\OpenCV资料\\模型资料\\灰度图彩色化模型\\model\\colorization_deploy_v2.prototxt";
    string modelBin = "H:\\资料\\学习资料\\视觉资料\\OpenCV资料\\模型资料\\灰度图彩色化模型\\model\\colorization_release_v2.caffemodel";
    //开始计时
    double t = (double)cv::getTickCount();
    // fixed input size for the pretrained network
    const int W_in = 224;
    const int H_in = 224;
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    // 设置训练得到的参数数据 在网络里额外添加两层
    int sz[] = { 2, 313, 1, 1 };
    //添加一个ab转换层
    const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
    Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
    class8_ab->blobs.push_back(pts_in_hull);
    //一个防止为输出为0的层
    Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
    conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));
    // 提取L通道灰度图，并均值化
    Mat lab, L, input;
    image.convertTo(image, CV_32F, 1.0 / 255);
    cvtColor(image, lab, COLOR_BGR2Lab);
    //提取亮度信息
    extractChannel(lab, L, 0);
    //重置大小
    resize(L, input, Size(W_in, H_in));
    input -= 50;
    // L通道图像输入到网络，前向计算
    Mat inputBlob = blobFromImage(input);
    net.setInput(inputBlob);
    Mat result = net.forward();
    // 从网络输出中提取得到的a,b通道
    Size siz(result.size[2], result.size[3]);
    //输出为56X56
    Mat a = Mat(siz, CV_32F, result.ptr(0, 0));
    Mat b = Mat(siz, CV_32F, result.ptr(0, 1));
    resize(a, a, image.size());
    resize(b, b, image.size());
    // 通道合并转换成彩色图
    Mat color, chn[] = { L, a, b };
    merge(chn, 3, lab);
    cvtColor(lab, color, COLOR_Lab2BGR);
    //计算时间
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Time taken : " << t << " secs" << endl;

    imshow("color", color);
}
//视频彩色化
void QuickDemo::video_colorization(String videoFileName)
{
    //创建VideoCapture对象，打开指定路径的视频文件
    VideoCapture cap(videoFileName);
    if (!cap.isOpened())
    {
        cerr << "Unable to open video" << endl;
        return;
    }
    //定义模型文件和权重文件的路径，以及用于存储视频帧的信息
    string protoFile = "H:\\资料\\学习资料\\视觉资料\\OpenCV资料\\模型资料\\灰度图彩色化模型\\model\\colorization_deploy_v2.prototxt";
    string weightsFile = "H:\\资料\\学习资料\\视觉资料\\OpenCV资料\\模型资料\\灰度图彩色化模型\\model\\colorization_release_v2.caffemodel";
    Mat frame, frameCopy;
    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    //生成输出视频文件的名称，并创建VideoWriter对象用于保存彩色化后的视频
    string str = videoFileName;
    str.replace(str.end() - 4, str.end(), "");
    string outVideoFileName = str + "_colorized.avi";
    VideoWriter video(outVideoFileName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, Size(frameWidth, frameHeight));

    //加载预训练的网络模型，并进行一些额外的设置
    const int W_in = 224;
    const int H_in = 224;
    Net net = dnn::readNetFromCaffe(protoFile, weightsFile);
    // setup additional layers:
    int sz[] = { 2, 313, 1, 1 };
    const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
    Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
    class8_ab->blobs.push_back(pts_in_hull);
    Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
    conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));

    //循环读取视频的每一帧，并对每一帧进行处理
    //将帧转换为 LAB 色彩空间，然后提取L通道并将其调整大小为网络所需的输入大小
    //对 L 通道进行了归一化，并通过网络传递以预测 a 和 b 通道
    //在获取预测的 a 和 b 通道后，它们被调整回原始帧大小
    //将 L、a 和 b 通道合并成一个 LAB 图像，然后将其转换回 BGR 色彩空间
    //将其写入输出视频文件中
    int i = 0;
    for (;;)
    {

        cap >> frame;
        if (frame.empty()) break;

        frameCopy = frame.clone();
        //将帧转换为 LAB 色彩空间，然后提取L通道并将其调整大小为网络所需的输入大小
        Mat lab, L, input;
        frame.convertTo(frame, CV_32F, 1.0 / 255);
        cvtColor(frame, lab, COLOR_BGR2Lab);
        extractChannel(lab, L, 0);
        resize(L, input, Size(W_in, H_in));
        input -= 50;
        //对 L 通道进行了归一化，并通过网络传递以预测 a 和 b 通道
        Mat inputBlob = blobFromImage(input);
        net.setInput(inputBlob);
        Mat result = net.forward();

        //在获取预测的 a 和 b 通道后，它们被调整回原始帧大小
        Size siz(result.size[2], result.size[3]);
        Mat a = Mat(siz, CV_32F, result.ptr(0, 0));
        Mat b = Mat(siz, CV_32F, result.ptr(0, 1));

        resize(a, a, frame.size());
        resize(b, b, frame.size());

        //将 L、a 和 b 通道合并成一个 LAB 图像，然后将其转换回 BGR 色彩空间
        Mat coloredFrame, chn[] = { L, a, b };
        merge(chn, 3, lab);
        cvtColor(lab, coloredFrame, COLOR_Lab2BGR);

        coloredFrame = coloredFrame * 255;
        coloredFrame.convertTo(coloredFrame, CV_8U);
        video.write(coloredFrame);
        imshow("视频彩色化演示", coloredFrame);
        i++;
        cout << "the current frame is: " << to_string(i) << "th" << endl;
    }
    cout << "Colorized video saved as " << outVideoFileName << endl << "Done !!!" << endl;
    cap.release();
    video.release();
}