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
    Range startX(150,  800), startY(550, 950);
    Range destX(170, 820), destY(550, 950);
    Detector detector(startX, startY, destX, destY, 2, 0);
    VideoCapture video("test.MOV");
    TwoMat frame;
    while (1) {
        char key = waitKey(500);
        if (key == 27) break;
        video >> frame.first;
        if (frame.first.empty()) {
            break;
        }
        video >> frame.second;
        if (frame.second.empty()) {
            break;
        }
        detector.pushFrame(frame);
        detector.detect(test);
    }
    return 0;
}
