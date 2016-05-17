#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include "src/DetectorManager.h"
#include <string>
#include <cstring>
#include <fstream>

using namespace cv;
using namespace std;

bool flag = 0;

void test(int startTag, int destTag, int carNum) {
    if (flag) {
        char start[3], dest[3];
        snprintf(start, 3, "%d", startTag);
        snprintf(dest, 3, "%d", destTag);

        std::string filePath("../forTest/text");
        filePath += start;
        filePath += "-";
        filePath += dest;
        filePath += ".txt";

        std::ofstream outfile(filePath, std::fstream::out|std::fstream::app);

        outfile << "Start: " << startTag << " ";
        outfile << "Dest: " << destTag << " ";
        outfile << "CarNum: " << carNum << endl;
        outfile.flush();
        outfile.close();
    } else {
        cout << "Start: " << startTag << " ";
        cout << "Dest: " << destTag << " ";
        cout << "CarNum: " << carNum << endl;
        cout.flush();
    }
}

int main() {
    Range startX(150,  800), startY(550, 950);
    Range destX(170, 820), destY(550, 950);
    //  set before get an instance
    DetectorManager::setDetectorManager(1, test);
    //  get an instance
    DetectorManager *manager = DetectorManager::getInstance();
    //  add areas
    manager->addStart(startX, startY, 1);
    manager->addDest(destX, destY, 1);
    manager->addStart(startX, startY, 2);
    manager->addDest(destX, destY, 2);
    //  start detecting job
    manager->start();
    VideoCapture video("../forTest/test.MOV");
    TwoMat frame;
    int i = 5;
    while (i > 0) {
        i--;
        char key = waitKey(1000);
        if (key == 27) break;
        video >> frame.first;
        if (frame.first.empty()) {
            break;
        }
        video >> frame.second;
        if (frame.second.empty()) {
            break;
        }
        //  keep pushing frame to the manager
        //  the detector will be waiting until it has enough frame to detect
        manager->pushFrame(frame);
        manager->pause();
        waitKey(2000);
        manager->resume();
    }
    //  terminate the detecting job
    manager->terminate();
    //  waiting for all sub-threads terminated is necessary
    //  or you may get an unexpected termination on your program
    waitKey(200);
    return 0;
}
