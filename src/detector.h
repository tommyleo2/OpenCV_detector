#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>

#define SPEED 10
#define MIN_DIS 0
#define MAX_DIS 50
#define MIN_DIF 5
#define MAX_DIF 20

using namespace cv;
using namespace std;

enum displayMode {LINE, POINT, ORIGIN, PAUSE};

class Detector {
public:
    Detector(const string &video = "");
    
    void detect();
private:
    VideoCapture cam;
    int countVehicle(vector<KeyPoint> &gK1, vector<KeyPoint> &gK2, vector<DMatch> &gMatches);
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
    char key = 'l';
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
        if (key == 'l') {
            mode = LINE;
        }
        if (key == 'p') {
            mode = POINT;
        }
        if (key == 'o') {
            mode = ORIGIN;
        }
        if (key == ' ') {
            mode = PAUSE;
        }

        if (mode == PAUSE) {
            key = cvWaitKey(0);
            continue;
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
            gK1.clear();
            gK2.clear();
            for (int i = 0; i < matches.size(); ++i) {
                //cout << "[" << i << "]" << "KeyPoint coordinate: " << keyPoints1[i].pt.x << " " << keyPoints1[i].pt.y << " | response: " << keyPoints1[i].response << " size: " << keyPoints1[i].size << endl;
                if (matches[i].distance < MAX_DIS && matches[i].distance > MIN_DIS) {  // pre-sift
                    float xDif = keyPoints1[matches[i].queryIdx].pt.x - keyPoints2[matches[i].trainIdx].pt.x, yDif = keyPoints1[matches[i].queryIdx].pt.y - keyPoints2[matches[i].trainIdx].pt.y;
                    xDif = xDif >= 0 ? xDif : (-1) * xDif;
                    yDif = yDif >= 0 ? yDif : (-1) * yDif;
                    cout << "xDif: " << xDif << ", yDif: " << yDif << endl;
                    if ((xDif > MIN_DIF && xDif < MAX_DIF) || (yDif > MIN_DIF && yDif < MAX_DIF)) {
                        
                        gMatches.push_back(matches[i]);
                        gK1.push_back(keyPoints1[i]);
                        gK2.push_back(keyPoints2[i]);
                    }

                    //int num = countVehicle(gK1, gK2, gMatches);
                    //cout << "[" << i << "]" << "KeyPoint coordinate: " << gK1[i].pt.x << " " << gK1[i].pt.y << " | response: " << gK1[i].response << " size: " << gK1[i].size << endl;
                    // cout << "Match[" << i << "] distance: " << matches[i].distance << " pushed" << endl;
                }
            }
            cout << "Total good points: " << gMatches.size() << " " << gK1.size() << " " << gK2.size() << endl;
            if (mode == LINE) {
                //drawMatches(pic1, keyPoints1, pic2, keyPoints2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                drawMatches(pic1, keyPoints1, pic2, keyPoints2, gMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            }
            if (mode == POINT) {
                drawKeypoints(pic1, gK1, imgPoint1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                drawKeypoints(pic2, gK2, imgPoint2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                drawMatches(imgPoint1, vector<KeyPoint>(), imgPoint2, vector<KeyPoint>(), vector<DMatch>(), imgMatches);
            }


        }
        imshow("Vehicle detector", imgMatches);
        key = cvWaitKey(1);
    }
}


int Detector::countVehicle(vector<KeyPoint> &gK1, vector<KeyPoint> &gK2, vector<DMatch> &gMatches) {
    
}
