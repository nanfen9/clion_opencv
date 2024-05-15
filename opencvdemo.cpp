//
// Created by HS-YF-1 on 2024/5/15.
//
#include <opencv2/dnn.hpp>
#include "opencvfunction.h"

using namespace std;
using namespace cv::dnn;

//ͼ��ɫ�ʿռ�ת��
void QuickDemo::colorSpace_Demo(Mat&image) {
    Mat gray, hsv;
    //ɫ�ʿռ�ת������ COLOR_BGR2GRAY=6 ��ɫת�Ҷ� COLOR_GRAY2BGR=8 �Ҷ�ת��ɫ
    //                 COLOR_BGR2HSV=40 BGRתHSV   COLOR_HSV2BGR=54 HSVתBGR
    cvtColor(image, hsv, COLOR_BGR2HSV);
    //H 0~180�� S,V����255
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("HSV", hsv);
    imshow("�Ҷ�", gray);
    //����ͼ��
    //imwrite("D:/hsv.bmp", hsv);
    //imwrite("D:/gray.png", gray);
}
//ͼ�����Ĵ����븳ֵ
void QuickDemo::mat_creation_demo(Mat& image) {
    Mat m1, m2;
    //��¡
    //m1 = image.clone();
    //����
    //image.copyTo(m2);
    //�����հ�ͼ��
    Mat m3 = Mat::zeros(Size(400, 400), CV_8UC3);
    Mat m4 = Mat::ones(Size(8, 8), CV_8UC1);
    //��ͼ��ֵ��ɫ
    m3 =Scalar(255,0,0);
    //��ʾ��Ϣ
    //std::cout << "Width:" << m3.cols << "heigth::" << m3.rows << "channels:" << m3.channels() << std::endl;
    //std::cout << m3 << std::endl;
    imshow("����ͼ��", m3);
    //��ֵ��
    Mat m5 = m3;
    m5 = Scalar(0,255,255);
    imshow("ͼ��", m3);
    //ֱ�Ӹ�ֵ����ͼ��
    Mat kernel = (Mat_ <char>(3, 3) << 0, -1, 0,
            -1, 5, -1,
            0, -1, 0
    );
    imshow("����ͼ��1", kernel);
}
//ͼ�����صĶ�д����
void QuickDemo::pixel_visit_demo(Mat&image) {
    int w = image.cols;
    int h = image.rows;
    int dims = image.channels();
    //for (int row=0;row<h;row++)
    //{
    //	for (int col=0;col<w;col++)
    //	{
    //		if (dims==1)//�Ҷ�ͼ��
    //		{
    //			int pv = image.at<uchar>(row, col);
    //			image.at<uchar>(row, col) = 255 - pv;
    //		}
    //		if (dims==3)//��ɫͼ��
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
            if (dims == 1)//�Ҷ�ͼ��
            {
                int pv = *current_row;
                *current_row++ = 255 - pv;
            }
            if (dims == 3)//��ɫͼ��
            {
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
            }
        }
    }
    imshow("���ض�д��ʾ", image);
}
//ͼ�����ص���������
void QuickDemo::operators_demo(Mat&image) {
    Mat dst= Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    m = Scalar(5, 5, 5);
    //�˷�����
    //multiply(image, m, dst);
    //�ӷ�����
    //dst = image + Scalar(50,50,50);

    //�ӷ�
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
    //�ӷ�
    //add(image, m, dst);
    //����
    //subtract(image, m, dst);
    //����
    divide(image, m, dst);

    imshow("��������", dst);
}
//
static void on_track(int b, void* userdata) {
    Mat image = *((Mat*)userdata);
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    //m = Scalar(b, b, b);
    //add(image, m, dst);
    addWeighted(image, 1.0, m, 0, b, dst);
    imshow("������Աȶȵ���", dst);
}
static void on_contrast(int b, void* userdata) {
    Mat image = *((Mat*)userdata);
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    double contrast = b / 100.0;
    addWeighted(image, contrast, m, 0.0, 0, dst);
    imshow("������Աȶȵ���", dst);
}
//����������-����ͼ�����ȺͶԱȶ�
void QuickDemo::tracking_bar_demo(Mat&image) {
    namedWindow("������Աȶȵ���", WINDOW_AUTOSIZE);
    int lightness = 50;
    int max_value = 100;
    int contrast_value = 100;
    //������
    createTrackbar("Value Bar:", "������Աȶȵ���", &lightness, max_value, on_track,(void*)(&image));
    createTrackbar("Contrast Bar:", "������Աȶȵ���", &contrast_value, 200, on_contrast, (void*)(&image));
    on_track(50, &image);
}
//�����¼�
void QuickDemo::key_demo(Mat&image) {
    Mat dst=Mat::zeros(image.size(),image.type());
    while (true)
    {
        //ÿ�εȴ�100����--����Ƶʱ��1
        int c = waitKey(100);
        if (c == 27) {
            break;
        }
        if (c==49)//Key#1
        {
            std::cout << "you enter key #1" << std::endl;
            cvtColor(image, dst, COLOR_BGR2GRAY);
            //imshow("������Ӧ", dst);
        }
        if (c == 50)//Key#2
        {
            std::cout << "you enter key #2" << std::endl;
            cvtColor(image, dst, COLOR_BGR2HSV);
            //imshow("������Ӧ", dst);
        }
        if (c == 51)//Key#3
        {
            std::cout << "you enter key #3" << std::endl;
            dst = Scalar(50, 50, 50);
            add(image, dst, dst);
        }
        imshow("������Ӧ", dst);
    }
}
//��ɫ���������
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
        //ÿ�εȴ�100����--����Ƶʱ��1
        int c = waitKey(2000);
        if (c == 27) {
            break;
        }
        //��ɫ���������
        applyColorMap(image, dst, colormap[index % 21]);
        index++;
        imshow("��ɫ�任", dst);
    }
}
//ͼ�����ص��߼�����
void QuickDemo::bitwise_demo(Mat&image) {
    //�����հ�ͼ��
    Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
    Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
    //���ƾ���
    rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
    rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
    imshow("m1", m1);
    imshow("m2", m2);
    Mat dst;
    //�����
    //bitwise_and(m1, m2, dst);
    //�����
    //bitwise_or(m1, m2, dst);
    //ȡ������-��
    //bitwise_not(image, dst);
    //dst = ~image;
    //������
    bitwise_xor(m1, m2, dst);
    imshow("����λ����", dst);
}
//ͼ��ͨ��������ϲ�
void QuickDemo::channels_demo(Mat&image) {
    //����
    std::vector<Mat>mv;
    //����
    split(image, mv);
    imshow("��ɫ", mv[0]);
    imshow("��ɫ", mv[1]);
    imshow("��ɫ", mv[2]);

    Mat dst;
    mv[1] = 0;
    mv[2] = 0;
    //�ϲ�
    merge(mv, dst);
    imshow("��ɫ", dst);
    //ͨ�����
    int from_to[] = { 0,2,1,1,2,0 };
    mixChannels(&image,1, &dst,1, from_to,3);
    imshow("ͨ�����", dst);
}
//ͼ��ɫ�ʿռ�ת��----��һ��ѧϰ
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
    //ͼ�񿽱�
    image.copyTo(redback, mask);
    imshow("ROI������ȡ", mask);
}
//ͼ������ֵͳ��
void QuickDemo::pixel_statistic_demo(Mat&image) {
    double minv, maxv;
    Point minloc, maxloc;
    //ͼ����Сֵ���ֵ����
    std::vector<Mat>mv;
    //����
    split(image, mv);
    for (int i=0;i<mv.size();i++)
    {
        minMaxLoc(mv[i], &minv, &maxv, &minloc, &maxloc, Mat());
        std::cout <<"No.channels"<<i<< "min value" << minv << "max value" << maxv << std::endl;
    }

    //ͼ���ֵ�ͷ���
    Mat mean, stddev;
    Mat redback = Mat::zeros(image.size(), image.type());
    redback = Scalar(40, 40, 200);
    imshow("redback", redback);
    meanStdDev(image, mean, stddev);
    mean.at<double>(1, 0);
    std::cout << "means:" << mean << "stddev:" << stddev << std::endl;
}
//ͼ�񼸺���״����
void QuickDemo::drawing_demo(Mat&image) {
    Rect rect;
    rect.x = 100;
    rect.y = 100;
    rect.width = 100;
    rect.height = 100;
    Mat bg = Mat::zeros(image.size(), image.type());
    //���ƾ���
    rectangle(bg, rect, Scalar(0, 0, 255), 3, 8, 0);
    //����Բ
    circle(bg, Point(150, 200), 50, Scalar(255, 0, 0), 3, 8, 0);
    //�����߶�
    line(bg, Point(100, 100), Point(200, 200), Scalar(0, 255, 0), 3, 8, 0);
    //������Բ
    RotatedRect rrt;
    rrt.center = Point(100, 100);
    rrt.size = Size(100, 150);
    rrt.angle = 0.0;
    ellipse(bg, rrt, Scalar(50, 255, 50), 3, 8);
    Mat dst;
    addWeighted(image, 0.7, bg, 0.3, 0, dst);
    imshow("������ʾ", dst);
}
//������������ɫ
void QuickDemo::random_drawing() {
    Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
    int w = canvas.cols;
    int h = canvas.rows;
    RNG rng(12345);
    while (true)
    {
        //ÿ�εȴ�100����--����Ƶʱ��1
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
        imshow("�������", canvas);
    }
}
//��������ͻ���
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
    //���ƶ��������
    fillPoly(canvas, pts, Scalar(255, 255, 0), 8, 0);
    //���ƶ��������
    polylines(canvas, pts, true, Scalar(0, 0, 255), 2, LINE_AA, 0);
    //���ƶ�������
    std::vector<std::vector<Point>>contours;
    contours.push_back(pts);
    drawContours(canvas, contours, -1, Scalar(0, 255, 0), 2);
    imshow("����λ���", canvas);
}
//����������Ӧ
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
            imshow("ROI����", image(box));
            rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
            imshow("������", image);
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
                imshow("������", image);
            }
        }
    }
}
void QuickDemo::mousse_drawing_demo(Mat&image) {
    namedWindow("������", WINDOW_AUTOSIZE);
    setMouseCallback("������", on_draw,(void*)(&image));
    imshow("������", image);
    temp = image.clone();
}
//ͼ����������ת�����һ��
void QuickDemo::norm_demo(Mat&image) {
    Mat dst;
    std::cout << image.type() << std::endl;
    //ͼ���ʽ�任--32λ
    image.convertTo(image, CV_32F);//CV_32S
    std::cout << image.type() << std::endl;
    normalize(image, dst, 1.0, 0,NORM_MINMAX);
    std::cout << dst.type() << std::endl;
    imshow("ͼ���һ��", dst);
}
//ͼ���������ֵ
void QuickDemo::resize_demo(Mat&image) {
    Mat zoomin, zoomout;
    int h = image.rows;
    int w = image.cols;
    resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
    imshow("zoomin", zoomin);
    resize(image, zoomout, Size(w * 1.5, h * 1.5), 0, 0, INTER_LINEAR);
    imshow("zoomout", zoomout);
}
//ͼ��ת
void QuickDemo::flip_demo(Mat&image) {
    Mat dst;
    flip(image, dst, 0);//0--���·�ת
    flip(image, dst, 1);//1--���ҷ�ת
    flip(image, dst, -1);//0--180����ת
    imshow("ͼ��ת", dst);
}
//ͼ����ת
void QuickDemo::rotate_demo(Mat&image) {
    Mat dst, M;
    int w = image.cols;
    int h = image.rows;
    //���ɾ���
    M = getRotationMatrix2D(Point(w / 2, h / 2), 85, 1.0);
    //��ȡ��ת��cos��sin
    double cos = abs(M.at<double>(0, 0));
    double sin = abs(M.at<double>(0, 1));
    //��ת��Ŀ�͸�
    int nw = cos * w + sin * h;
    int nh = sin * w + cos * h;
    //��ת�������
    M.at<double>(0, 2) = M.at<double>(0, 2) + (nw / 2 - w / 2);
    M.at<double>(1, 2) = M.at<double>(1, 2) + (nh / 2 - h / 2);
    //ͼ����ת
    warpAffine(image, dst, M, Size(nw,nh),INTER_LINEAR,0,Scalar(255,0,0));
    imshow("��ת��ʾ", dst);
}
//��Ƶ�ļ�����ͷʹ�ú���Ƶ�����뱣��
void QuickDemo::video_demo(Mat&image) {
    //��ȡ��Ƶ�ļ�������ͷ
    VideoCapture capture("D:/���/����ͼƬ/����.mp4");
    //��ȡ��Ƶ�Ĳ���
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(CAP_PROP_FRAME_COUNT);
    double fps = capture.get(CAP_PROP_FPS);
    //��ʾ
    std::cout << "frame_width:" << frame_width << std::endl;
    std::cout << "frame_height:" << frame_height << std::endl;
    std::cout << "Number of Frames:" << count << std::endl;
    std::cout << "FPS:" << fps << std::endl;
    //�����������Ƶ·���͸�ʽ
    VideoWriter writer("D:/���/����ͼƬ/2.mp4",capture.get(CAP_PROP_FOURCC),fps,Size(frame_width, frame_height),true);
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
        //������Ƶ
        writer.write(frame);
        int c = waitKey(10);
        if (c == 27) {
            break;
        }
    }
    //�����Ƶ�ڴ�
    capture.release();
    writer.release();
}
//ͼ��ֱ��ͼ
void QuickDemo::histogram_demo(Mat&image) {
    //��ͨ������
    std::vector<Mat>bgr_plane;
    split(image, bgr_plane);
    //�����������
    const int channels[1] = { 0 };
    const int bins[1] = { 256 };
    float hranges[2] = { 0,255 };
    const float* ranges[1] = { hranges };
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    //����Blue��green��redͨ����ֱ��ͼ
    calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
    //��ʾֱ��ͼ
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / bins[0]);
    Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
    //��һ��ֱ��ͼ����
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    //����ֱ��ͼ����
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
//��άֱ��ͼ
void QuickDemo::histogram_2d_demo(Mat&image) {
    //2Dֱ��ͼ
    Mat hsv, hs_hist;
    //ת����hsvͨ��ͼ��
    cvtColor(image, hsv, COLOR_BGR2HSV);
    //�������
    int hbins = 30, sbins = 32;
    int hist_bins[] = { hbins,sbins };
    float h_range[] = { 0,180 };
    float s_range[] = { 0,256 };
    const float* hs_ranges[] = { h_range,s_range };
    int hs_channels[] = { 0,1 };
    //��ȡͼ��Ķ�άֱ��ͼ
    calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
    double maxVal = 0;
    minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
    int scale = 10;
    Mat hist2d_image = Mat::zeros(sbins*scale, hbins*scale, CV_8UC3);
    //���ƶ�άֱ��ͼ
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
//ֱ��ͼ���⻯
void QuickDemo::histogram_eq_demo(Mat&image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat dst;
    //ֱ��ͼ���⻯
    equalizeHist(gray, dst);
    imshow("ֱ��ͼ���⻯��ʾ", dst);
}
//ͼ��������
void QuickDemo::blur_demo(Mat&image) {
    Mat dst;
    blur(image, dst, Size(3, 3), Point(-1, -1));
    imshow("ͼ��ģ��", dst);
}
//��˹ģ��
void QuickDemo::gaussian_blur_demo(Mat&image) {
    Mat dst;
    GaussianBlur(image, dst, Size(5, 5), 15);
    imshow("��˹ģ��", dst);
}
//��˹˫��ģ��
void QuickDemo::bifilter_demo(Mat&image) {
    Mat dst;
    bilateralFilter(image, dst, 0, 100, 10);
    imshow("˫��ģ��", dst);
}
//ʵʱ�������
void QuickDemo::face_detection_demo(){
    //����ģ��·��
    string root_dir = "E:/OpenCV/OpenCV4.8/opencv/sources/samples/dnn/face_detector/";
    //����ģ��·��
    Net net=readNetFromTensorflow(root_dir + "opencv_face_detector_uint8.pb", root_dir + "opencv_face_detector.pbtxt");
    //��ȡ��Ƶ��ͼ���ļ�
    VideoCapture capture(0);
    Mat frame;
    while (true)
    {
        //����ȡ���ļ��浽����
        capture.read(frame);

        flip(frame, frame, 1);
        if (frame.empty())
        {
            break;
        }
        //����ģ�Ͳ���
        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
        //����ģ��
        net.setInput(blob);//NCHW
        //ģ������
        Mat probs = net.forward();//
        Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
        //�������
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
        imshow("���������ʾ", frame);
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
}
//ͼ��Ǧ�ʻ��Ϳ�ͨ��
void QuickDemo::image_cartoonization_demo(Mat& image)
{
    Mat imgGray, imgColor;
    //��ɫǦ�ʻ�����ɫǦ�ʻ�Ч��
    pencilSketch(image, imgGray, imgColor, 5, 0.1f, 0.03f);
    Mat result;
    //��ͨ��Ч��
    stylization(image, result, 5, 0.6);
    imshow("��ɫǦ�ʻ�Ч��", imgGray);
    imshow("��ɫǦ�ʻ�Ч��", imgColor);
    imshow("��ͨ��Ч��", result);
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
//�Ҷ�ͼ��ɫ��
void QuickDemo::grayscale_is_colored(Mat& image)
{

    string modelTxt = "H:\\����\\ѧϰ����\\�Ӿ�����\\OpenCV����\\ģ������\\�Ҷ�ͼ��ɫ��ģ��\\model\\colorization_deploy_v2.prototxt";
    string modelBin = "H:\\����\\ѧϰ����\\�Ӿ�����\\OpenCV����\\ģ������\\�Ҷ�ͼ��ɫ��ģ��\\model\\colorization_release_v2.caffemodel";
    //��ʼ��ʱ
    double t = (double)cv::getTickCount();
    // fixed input size for the pretrained network
    const int W_in = 224;
    const int H_in = 224;
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    // ����ѵ���õ��Ĳ������� ������������������
    int sz[] = { 2, 313, 1, 1 };
    //���һ��abת����
    const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
    Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
    class8_ab->blobs.push_back(pts_in_hull);
    //һ����ֹΪ���Ϊ0�Ĳ�
    Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
    conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));
    // ��ȡLͨ���Ҷ�ͼ������ֵ��
    Mat lab, L, input;
    image.convertTo(image, CV_32F, 1.0 / 255);
    cvtColor(image, lab, COLOR_BGR2Lab);
    //��ȡ������Ϣ
    extractChannel(lab, L, 0);
    //���ô�С
    resize(L, input, Size(W_in, H_in));
    input -= 50;
    // Lͨ��ͼ�����뵽���磬ǰ�����
    Mat inputBlob = blobFromImage(input);
    net.setInput(inputBlob);
    Mat result = net.forward();
    // �������������ȡ�õ���a,bͨ��
    Size siz(result.size[2], result.size[3]);
    //���Ϊ56X56
    Mat a = Mat(siz, CV_32F, result.ptr(0, 0));
    Mat b = Mat(siz, CV_32F, result.ptr(0, 1));
    resize(a, a, image.size());
    resize(b, b, image.size());
    // ͨ���ϲ�ת���ɲ�ɫͼ
    Mat color, chn[] = { L, a, b };
    merge(chn, 3, lab);
    cvtColor(lab, color, COLOR_Lab2BGR);
    //����ʱ��
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Time taken : " << t << " secs" << endl;

    imshow("color", color);
}
//��Ƶ��ɫ��
void QuickDemo::video_colorization(String videoFileName)
{
    //����VideoCapture���󣬴�ָ��·������Ƶ�ļ�
    VideoCapture cap(videoFileName);
    if (!cap.isOpened())
    {
        cerr << "Unable to open video" << endl;
        return;
    }
    //����ģ���ļ���Ȩ���ļ���·�����Լ����ڴ洢��Ƶ֡����Ϣ
    string protoFile = "H:\\����\\ѧϰ����\\�Ӿ�����\\OpenCV����\\ģ������\\�Ҷ�ͼ��ɫ��ģ��\\model\\colorization_deploy_v2.prototxt";
    string weightsFile = "H:\\����\\ѧϰ����\\�Ӿ�����\\OpenCV����\\ģ������\\�Ҷ�ͼ��ɫ��ģ��\\model\\colorization_release_v2.caffemodel";
    Mat frame, frameCopy;
    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    //���������Ƶ�ļ������ƣ�������VideoWriter�������ڱ����ɫ�������Ƶ
    string str = videoFileName;
    str.replace(str.end() - 4, str.end(), "");
    string outVideoFileName = str + "_colorized.avi";
    VideoWriter video(outVideoFileName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, Size(frameWidth, frameHeight));

    //����Ԥѵ��������ģ�ͣ�������һЩ���������
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

    //ѭ����ȡ��Ƶ��ÿһ֡������ÿһ֡���д���
    //��֡ת��Ϊ LAB ɫ�ʿռ䣬Ȼ����ȡLͨ�������������СΪ��������������С
    //�� L ͨ�������˹�һ������ͨ�����紫����Ԥ�� a �� b ͨ��
    //�ڻ�ȡԤ��� a �� b ͨ�������Ǳ�������ԭʼ֡��С
    //�� L��a �� b ͨ���ϲ���һ�� LAB ͼ��Ȼ����ת���� BGR ɫ�ʿռ�
    //����д�������Ƶ�ļ���
    int i = 0;
    for (;;)
    {

        cap >> frame;
        if (frame.empty()) break;

        frameCopy = frame.clone();
        //��֡ת��Ϊ LAB ɫ�ʿռ䣬Ȼ����ȡLͨ�������������СΪ��������������С
        Mat lab, L, input;
        frame.convertTo(frame, CV_32F, 1.0 / 255);
        cvtColor(frame, lab, COLOR_BGR2Lab);
        extractChannel(lab, L, 0);
        resize(L, input, Size(W_in, H_in));
        input -= 50;
        //�� L ͨ�������˹�һ������ͨ�����紫����Ԥ�� a �� b ͨ��
        Mat inputBlob = blobFromImage(input);
        net.setInput(inputBlob);
        Mat result = net.forward();

        //�ڻ�ȡԤ��� a �� b ͨ�������Ǳ�������ԭʼ֡��С
        Size siz(result.size[2], result.size[3]);
        Mat a = Mat(siz, CV_32F, result.ptr(0, 0));
        Mat b = Mat(siz, CV_32F, result.ptr(0, 1));

        resize(a, a, frame.size());
        resize(b, b, frame.size());

        //�� L��a �� b ͨ���ϲ���һ�� LAB ͼ��Ȼ����ת���� BGR ɫ�ʿռ�
        Mat coloredFrame, chn[] = { L, a, b };
        merge(chn, 3, lab);
        cvtColor(lab, coloredFrame, COLOR_Lab2BGR);

        coloredFrame = coloredFrame * 255;
        coloredFrame.convertTo(coloredFrame, CV_8U);
        video.write(coloredFrame);
        imshow("��Ƶ��ɫ����ʾ", coloredFrame);
        i++;
        cout << "the current frame is: " << to_string(i) << "th" << endl;
    }
    cout << "Colorized video saved as " << outVideoFileName << endl << "Done !!!" << endl;
    cap.release();
    video.release();
}