#include <opencv2/opencv.hpp>
#include <vector>
#include "detector.h"

using namespace cv;
using namespace std;

void test(int carNum, Mat pic) {
    //cout << carNum << endl;
    //imshow(",,,", pic);
}

int main() {
    //Range startX(100,  350), startY(550, 950);
    //Range destX(950, 1200), destY(600, 1000);
    Range startX(150,  400), startY(550, 950);
    Range destX(170, 420), destY(550, 950);
    Detector detector(startX, startY, destX, destY, 0, 0);
    VideoCapture video("test.MOV");
    Mat frame1, frame2;
    while (1) {
        char key = cvWaitKey(100);
        if (key == 27) break;
        video >> frame1;
        if (frame1.empty()) {
            break;
        }
        video >> frame2;
        if (frame2.empty()) {
            break;
        }
        detector.detect(frame1, frame2, test);
    }
    return 0;
}
