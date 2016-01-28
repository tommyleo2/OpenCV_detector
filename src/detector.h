#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>

#define SPEED 3
#define MIN_DIS 30
#define MAX_DIS 50

using namespace cv;
using namespace std;

enum displayMode {LINE, POINT, ORIGIN};

class Detector {
public:
    Detector(const string &video = "");
    
    void detect();
private:
    VideoCapture cam;
};

Detector::Detector(const string &video) {
    if (video != "") {
        cam.open(video);
    }
}

void Detector::detect() {
    Mat pic1, pic2, frame;
    Ptr<ORB> detector = ORB::create();
    vector<KeyPoint> keyPoints1, keyPoints2, gK1, gK2;
    Mat discriptor1, discriptor2;
    vector<DMatch> matches, gMatches;
    Ptr<DescriptorMatcher> matcher;
    Mat imgMatches, imgPoint1, imgPoint2;
    bool toggle = true;  // determine which of the pic1 and pic2 is the previous picture
    displayMode mode = LINE;  // switch between display modes
    cam >> pic1;
    while (1) {

        for (int i = 0; i < SPEED; ++i) { // catch frame
            if (toggle) {
                cam >> pic2;
                if (pic2.empty()) {
                    return;
                }

            } else {
                cam >> pic1;
                if (pic1.empty()) {
                    return;
                }
            }
        }
        toggle = !toggle;
        char key = cvWaitKey(1);
        if (key == 'l') {
            mode = LINE;
        }
        if (key == 'p') {
            mode = POINT;
        }
        if (key == 'o') {
            mode = ORIGIN;
        }

        if (mode == ORIGIN) {
            drawMatches(pic1, vector<KeyPoint>(), pic2, vector<KeyPoint>(), vector<DMatch>(), imgMatches);
        } else {
            detector->detect(pic1, keyPoints1, Mat());
            detector->compute(pic1, keyPoints1, discriptor1);
            detector->detect(pic2, keyPoints2, Mat());
            detector->compute(pic2, keyPoints2, discriptor2);

            matcher = DescriptorMatcher::create("BruteForce-Hamming");
            matcher->match(discriptor1, discriptor2, matches);

            gMatches.clear();
            for (int i = 0; i < discriptor1.rows; ++i) {
                if (matches[i].distance < MAX_DIS && matches[i].distance > MIN_DIS) {
                    gMatches.push_back(matches[i]);
                    //                    gK1.push_back(keyPoints1[i]);
                    //                    gK2.push_back(keyPoints2[i]);
                    // cout << "Match[" << i << "] distance: " << matches[i].distance << " pushed" << endl;
                }
            }
            
            if (mode == LINE) {
                cout << "Total good points: " << gMatches.size() << endl;
                //drawMatches(pic1, keyPoints1, pic2, keyPoints2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                drawMatches(pic1, keyPoints1, pic2, keyPoints2, gMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            }

        }
        imshow("Vehicle detector", imgMatches);
    }
}
