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
    Range startX(100,  350), startY(550, 950);
    Range destX(950, 1200), destY(600, 1000);
    Detector detector(startX, startY, destX, destY);
    VideoCapture video("test.MOV");
    Mat frame;
    while (1) {
        char key = cvWaitKey(00);
        if (key == 27) break;
        video >> frame;
        if (frame.empty()) {
            break;
        }
        detector.detect(frame, test, 1);
    }
    return 0;
}
