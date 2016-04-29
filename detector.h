#ifndef DETECTOR_H
#define DETECTOR_H

/*
  Usage:
    1. Get a instance of the Detector
    2. Create a thread of detect(void (*showNum)(int)) to perform the detecting
    3. Keep pushing new frame to the rawFramePool

  Note:
    1. The origin point is at the left upper of a picture
    2. Parameter _startX and _destX indicate the range of column of a Mat.
    3. Parameter _startY and _destX indicate the range of row of a Mat.
 */


#include <opencv2/opencv.hpp>
#include <iterator>
#include <vector>
#include <list>
#include <queue>

#define CARDETECTED 20
#define MAXDIFF 10

using namespace cv;
using namespace std;

class Detector {
public:
    Detector(Range _startX, Range _startY, Range _destX, Range _destY);
    void setStart(Range _startX, Range _startY);
    void setDest(Range _destX, Range _destY);
    void detect(Mat frame, void (*showNum)(int, Mat), bool debug = 0);
    void pushFrame(Mat frame);
private:
    void detectKeyPoints(Mat start, Mat dest, bool debug = 0);
    void matchAndDelete(bool debug = 0);
    void findCar(vector<DMatch> &matches);
    queue<Mat> rawFramePool;
    list< vector<KeyPoint> > startKeyPoints;
    vector<KeyPoint> destKeyPoints;
    list<Mat> startDescriptor;
    Mat destDescriptor;
    Range startX, startY, destX, destY;
    int carNum;
    Ptr<DescriptorMatcher> matcher;
    //int totalNum;
};

void Detector::pushFrame(Mat frame) {
    rawFramePool.push(frame);
}

Detector::Detector(Range _startX, Range _startY, Range _destX, Range _destY) {
    startX = _startX;
    startY = _startY;
    destX = _destX;
    destY = _destY;
    carNum = 0;
    matcher = DescriptorMatcher::create("BruteForce-Hamming");
}

void Detector::setStart(Range _startX, Range _startY) {
    startX = _startX;
    startY = _startY;
}

void Detector::setDest(Range _destX, Range _destY) {
    destX = _destX;
    destY = _destY;
}

void Detector::detect(Mat frame, void (*showNum)(int, Mat), bool debug) {

    Mat start = frame(startY, startX);
    Mat dest = frame(destY, destX);
    Mat imgMatches;
    detectKeyPoints(start, dest, debug);
    //matchAndDelete(debug);
    //showNum(carNum, imgMatches);
}

void Detector::detectKeyPoints(Mat start, Mat dest, bool debug) {
    vector<KeyPoint> keyPoints1;
    Mat descriptor1;
    Ptr<ORB> detector = ORB::create();
    detector->detect(start, keyPoints1, Mat());
    detector->compute(start, keyPoints1, descriptor1);
    detector->detect(dest, destKeyPoints, Mat());
    detector->compute(dest, destKeyPoints, destDescriptor);
    startKeyPoints.push_back(keyPoints1);
    startDescriptor.push_back(descriptor1);
    if (debug) {
        vector<DMatch> matches;
        Mat imgMatch;
        matcher->match(descriptor1, destDescriptor, matches);
        drawMatches(start, keyPoints1, dest, destKeyPoints, matches, imgMatch, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cout << "keyPoints: " << keyPoints1.size() << " " << destKeyPoints.size() << endl;
        cout << "matches: " << matches.size() << endl;
        namedWindow("Debug", WINDOW_NORMAL);
        imshow("Debug", imgMatch);
    }
}

void Detector::matchAndDelete(bool debug) {
    vector<DMatch> matches;
    
    for (list<Mat>::iterator it = startDescriptor.begin(); it != startDescriptor.end(); it++) {
        matcher->match(*it, destDescriptor, matches);

        if (matches.size() > CARDETECTED) {
            findCar(matches);
        }
        matches.clear();
    }
    if (startDescriptor.size() > 10) {
        startDescriptor.pop_front();
        startKeyPoints.pop_front();
    }
}

void Detector::findCar(vector<DMatch> &matches) {
    vector <vector<DMatch> > vehicle;
    bool *mark = new bool[matches.size()];
    for (int i = 0; i < matches.size(); ++i) {
        mark[i] = true;
    }

    int circleNum = 0;
    for (int i = 0; i < matches.size(); ++i) {
        if (mark[i] == false) {
            continue;
        }
        vehicle.push_back(vector<DMatch>());
        vehicle[circleNum].push_back(matches[i]);
        //  cout << vehicle.size() << " " << vehicle[circleNum].size() << endl;
        mark[i] = false;
        for (int j = 1; j < matches.size(); ++j) {
            if (mark[j] == false) {
                continue;
            }
            for (int k = 0; k < vehicle[circleNum].size(); ++k) {
                float xDif = destKeyPoints[vehicle[circleNum][k].trainIdx].pt.x - destKeyPoints[matches[j].trainIdx].pt.x,
                      yDif = destKeyPoints[vehicle[circleNum][k].trainIdx].pt.y - destKeyPoints[matches[j].trainIdx].pt.y;
                xDif = xDif > 0 ? xDif : xDif * (-1);
                yDif = yDif > 0 ? yDif : yDif * (-1);
                if (xDif < MAXDIFF && yDif < MAXDIFF) {
                    // cout << circleNum << " " <<  vehicle[circleNum].size() << " " << j << endl;
                    vehicle[circleNum].push_back(matches[j]);
                    mark[j] = false;
                    j = 1;
                    break;
                }
            }
        }
        if (vehicle[circleNum].size() > 5) {
            carNum++;
        }
        circleNum++;
        //cout << circleNum << " " << num << endl;
    }
}

#endif /* DETECTOR_H */
