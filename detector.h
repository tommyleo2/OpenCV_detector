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
#define MAX_DIS 100 //distance in DMatch
#define MAX_DIFF 30
#define MIN_DIFF 5   //KeyPoints distance
#define MAX_RANGE 15 //maximum distance of keypoints in the same vehicle
#define MIN_CLUSTER 5 //minimun number of keypoints to define a car
#define RETAIN_TIME 10 //each frame will be retained for 10s in pool

using namespace cv;
using namespace std;

typedef struct {
    Mat first, second;
} TwoMat;

typedef struct {
    vector<KeyPoint> kp;
    Mat img;
} Start;

class Detector {
public:
    Detector(Range _startX, Range _startY, Range _destX, Range _destY, int _fps, bool _debug = false);
    void setOriginStart(Mat _originStart);
    void setOriginDest(Mat _originDest);
    void setStart(Range _startX, Range _startY);
    void setDest(Range _destX, Range _destY);
    void setMaxPoolSize(int _fps);
    void detect(void (*showNum)(int));
    void pushFrame(TwoMat frame);
private:
    void detect_compute(Mat &frame, vector<KeyPoint> &kp, Mat &dp);
    int findVehicle(vector<KeyPoint> &kp1, Mat &dp1, vector<KeyPoint> &kp2, Mat &dp2);
    int matchVehicle(Mat &frame2, vector<KeyPoint> &kp2);

    queue<TwoMat> rawFramePool;
    list<Start> startList;
    Range startX, startY, destX, destY;
    int carNum;
    Ptr<ORB> detector;
    Ptr<DescriptorMatcher> matcher;
    //int totalNum;

    int maxPoolSize;
    bool debug;
};

void Detector::pushFrame(TwoMat frame) {
    rawFramePool.push(frame);
}

Detector::Detector(Range _startX, Range _startY, Range _destX, Range _destY, int _fps, bool _debug) {
    startX = _startX;
    startY = _startY;
    destX = _destX;
    destY = _destY;
    carNum = 0;
    maxPoolSize = _fps * RETAIN_TIME;
    matcher = DescriptorMatcher::create("BruteForce-Hamming");
    detector = ORB::create();
    debug = _debug;
    if (debug) {
        namedWindow("debug", WINDOW_NORMAL);
    }
}

void Detector::setStart(Range _startX, Range _startY) {
    startX = _startX;
    startY = _startY;
}

void Detector::setDest(Range _destX, Range _destY) {
    destX = _destX;
    destY = _destY;
}

void Detector::setMaxPoolSize(int _fps) {
    maxPoolSize = _fps * RETAIN_TIME;
}

void Detector::detect(void (*showNum)(int)) {
    while (rawFramePool.empty());
    TwoMat frame = rawFramePool.front();
    rawFramePool.pop();
    Mat start1 = (frame.first)(startY, startX);
    Mat dest1 = (frame.first)(destY, destX);
    Mat start2 = (frame.second)(startY, startX);
    Mat dest2 = (frame.second)(destY, destX);
    vector<KeyPoint> start1KeyPoints, start2KeyPoints, dest1KeyPoints, dest2KeyPoints;
    Mat start1Descriptor, start2Descriptor, dest1Descriptor, dest2Descriptor;
    detect_compute(start1, start1KeyPoints, start1Descriptor);
    detect_compute(start2, start2KeyPoints, start2Descriptor);
    detect_compute(dest1, dest1KeyPoints, dest1Descriptor);
    detect_compute(dest2, dest2KeyPoints, dest2Descriptor);
    if (debug) {
        cout << "origin:\n";
        cout << "start2.KeyPoints.size() = " << start2KeyPoints.size() << endl;
        cout << "dest2.KeyPoints.size() = " << dest2KeyPoints.size() << endl;
        cout << "---\n";
    }
    int startNum = findVehicle(start1KeyPoints, start1Descriptor, start2KeyPoints, start2Descriptor);
    int destNum = findVehicle(dest1KeyPoints, dest1Descriptor, dest2KeyPoints, dest2Descriptor);
    Start currentStart;
    currentStart.kp = start2KeyPoints;
    currentStart.img = start2;
    startList.push_back(currentStart);
    if (startList.size() >= maxPoolSize) {
        startList.pop_front();
    }
    if (debug) {
        Mat imgWithPoint;
        cout << "start2.KeyPoints.size() = " << start2KeyPoints.size() << endl;
        drawKeypoints(start2, start2KeyPoints, imgWithPoint);
        imshow("debug", imgWithPoint);
        waitKey(0);
        cout << "dest2.KeyPoints.size() = " << dest2KeyPoints.size() << endl;
        drawKeypoints(dest2, dest2KeyPoints, imgWithPoint);
        imshow("debug", imgWithPoint);
        waitKey(0);
    }
    int matchNum = matchVehicle(dest2, dest2KeyPoints);
    //cout << matchNum << endl;
    showNum(matchNum);
}

void Detector::detect_compute(Mat &frame, vector<KeyPoint> &kp, Mat &dp) {
    detector->detect(frame, kp, Mat());
    detector->compute(frame, kp, dp);
}

int Detector::findVehicle(vector<KeyPoint> &kp1, Mat &dp1, vector<KeyPoint> &kp2, Mat &dp2) {
    vector<DMatch> matches;
    int count = 0;
    matcher->match(dp2, dp1, matches);
    priority_queue<int> deleteQueue;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance < MAX_DIS) {
            float xDiff = kp2[matches[i].queryIdx].pt.x - kp1[matches[i].trainIdx].pt.x, yDiff = kp2[matches[i].queryIdx].pt.y - kp1[matches[i].trainIdx].pt.y;
            xDiff = xDiff >= 0 ? xDiff : (-1) * xDiff;
            yDiff = yDiff >= 0 ? yDiff : (-1) * yDiff;
            if (!(xDiff > MIN_DIFF && xDiff < MAX_DIFF && yDiff > MIN_DIFF && yDiff < MAX_DIFF)) {
                deleteQueue.push(matches[i].queryIdx);
                //                kp2.erase(kp2.begin() + matches[i].queryIdx);
            } else {
                count++;
            }
        } else {
            //            kp2.erase(kp2.begin() + matches[i].queryIdx);
            deleteQueue.push(matches[i].queryIdx);
        }
    }
    while (!deleteQueue.empty()) {
        kp2.erase(kp2.begin() + deleteQueue.top());
        deleteQueue.pop();
    }
    return count;
}

int Detector::matchVehicle(Mat &frame2, vector<KeyPoint> &kp2) {
    Mat frame1;
    vector<KeyPoint> kp1;
    Mat dp1, dp2;
    vector<DMatch> matches;
    int count = 0;
    for (auto it = startList.begin(); it != startList.end(); it++) {
        frame1 = it->img;
        kp1 = it->kp;
        detector->compute(frame1, kp1, dp1);
        detector->compute(frame2, kp2, dp2);
        matcher->match(dp2, dp1, matches);


        bool *mark = new bool[matches.size()];
        for (int i = 0; i < matches.size(); i++) {
            mark[i] = false;
        }
        for (int i = 0; i < matches.size(); i++) {
            if (mark[i]) {
                continue;
            }
            mark[i] = true;
            vector<DMatch> cluster;
            cluster.push_back(matches[i]);
            if (debug) {
                cout << "--------\nkpNum:\n";
            }
            for (int j = 1 + i; j < matches.size(); j++) {
                if (mark[j]) {
                    continue;
                }
                if (debug) {
                    cout << cluster.size() << endl;
                }
                for (int k = 0; k < cluster.size(); k++) {
                    float dxdiff = kp2[matches[j].queryIdx].pt.x - kp2[cluster[k].queryIdx].pt.x;
                    float dydiff = kp2[matches[j].queryIdx].pt.y - kp2[cluster[k].queryIdx].pt.y;
                    float sxdiff = kp1[matches[j].trainIdx].pt.x - kp1[cluster[k].trainIdx].pt.x;
                    float sydiff = kp1[matches[j].trainIdx].pt.y - kp1[cluster[k].trainIdx].pt.y;
                    sxdiff = sxdiff >= 0 ? sxdiff : (-1) * sxdiff;
                    sydiff = sydiff >= 0 ? sydiff : (-1) * sydiff;
                    dxdiff = dxdiff >= 0 ? dxdiff : (-1) * dxdiff;
                    dydiff = dydiff >= 0 ? dydiff : (-1) * dydiff;
                
                    if (sxdiff < MAX_RANGE && sydiff < MAX_RANGE && dxdiff < MAX_RANGE && dydiff < MAX_RANGE) {
                        mark[j] = true;
                        cluster.push_back(matches[j]);
                        if (debug) {
                            Mat result;
                            cout << "sxdiff: " << sxdiff << endl;
                            cout << "dxdiff: " << dxdiff << endl;
                            cout << "sydiff: " << sydiff << endl;
                            cout << "dydiff: " << dydiff << endl;
                            cout << kp2[matches[j].queryIdx].pt.x << " " << kp2[matches[j].queryIdx].pt.y << endl;
                            cout << kp2[cluster[k].queryIdx].pt.x << " " << kp2[cluster[k].queryIdx].pt.y << endl;
                            cout << kp1[matches[j].queryIdx].pt.x << " " << kp1[matches[j].queryIdx].pt.y << endl;
                            cout << kp1[cluster[k].queryIdx].pt.x << " " << kp1[cluster[k].queryIdx].pt.y << endl;
                            drawMatches(frame2, kp2, frame1, kp1, cluster, result, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                            imshow("debug", result);
                            waitKey(0);
                            
                        }
                        j = i;
                        break;
                    }
                }
            }
            if (debug) {
                Mat result;
                drawMatches(frame2, kp2, frame1, kp1, cluster, result, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                imshow("debug", result);
                waitKey(0);
                cout << "------\n";
            }
            if (cluster.size() > MIN_CLUSTER) {
                count++;
            }
        }
        if (debug) {
            Mat result;
            drawMatches(frame2, kp2, frame1, kp1, matches, result, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            imshow("debug", result);
            waitKey(0);
        }
        delete[] mark;
    }
    
    return count;
}

#endif /* DETECTOR_H */
