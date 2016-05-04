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

using namespace cv;
using namespace std;

class Detector {
public:
    Detector(Range _startX, Range _startY, Range _destX, Range _destY, int _fps, bool _debug = false);
    void setOriginStart(Mat _originStart);
    void setOriginDest(Mat _originDest);
    void setStart(Range _startX, Range _startY);
    void setDest(Range _destX, Range _destY);
    void setFps(int _fps);
    void detect(Mat frame1, Mat frame2, void (*showNum)(int, Mat));
    void pushFrame(Mat frame);
private:
    void detect_compute(Mat &frame, vector<KeyPoint> &kp, Mat &dp);
    int findVehicle(vector<KeyPoint> &kp1, Mat &dp1, vector<KeyPoint> &kp2, Mat &dp2);

    
    void detectKeyPoints(Mat start, Mat dest);
    void matchAndDelete();
    void findCar(vector<DMatch> &matches);
    int matchVehicle(Mat &frame1, vector<KeyPoint> &kp1, Mat &frame2, vector<KeyPoint> &kp2);
    
    queue<Mat> rawFramePool;
    list< vector<KeyPoint> > startKeyPoints;
    vector<KeyPoint> destKeyPoints;
    list<Mat> startDescriptor;
    Mat destDescriptor;
    Range startX, startY, destX, destY;
    int carNum;
    Ptr<ORB> detector;
    Ptr<DescriptorMatcher> matcher;
    //int totalNum;
    vector<KeyPoint> originStartKeyPoints, originDestKeyPoints;
    Mat originStartDescriptor, originDestDescriptor;
    int fps;  //Rate of pushing frame, Frame per second
    bool debug;
};

void Detector::pushFrame(Mat frame) {
    rawFramePool.push(frame);
}

Detector::Detector(Range _startX, Range _startY, Range _destX, Range _destY, int _fps, bool _debug) {
    startX = _startX;
    startY = _startY;
    destX = _destX;
    destY = _destY;
    carNum = 0;
    fps = _fps;
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

void Detector::setOriginStart(Mat _originStart) {
    detector->detect(_originStart, originStartKeyPoints, Mat());
    detector->compute(_originStart, originStartKeyPoints, originStartDescriptor);
}

void Detector::setOriginDest(Mat _originDest) {
    detector->detect(_originDest, originDestKeyPoints, Mat());
    detector->compute(_originDest, originDestKeyPoints, originDestDescriptor);
}

void Detector::setFps(int _fps) {
    fps = _fps;
}

void Detector::detect(Mat frame1, Mat frame2, void (*showNum)(int, Mat)) {

    Mat start1 = frame1(startY, startX);
    Mat dest1 = frame1(destY, destX);
    Mat start2 = frame2(startY, startX);
    Mat dest2 = frame2(destY, destX);
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
    int matchNum = matchVehicle(start2, start2KeyPoints, dest2, dest2KeyPoints);
    cout << matchNum << endl;
    //Mat imgMatches;


    //matchAndDelete(debug);
    //showNum(carNum, imgMatches);
}

void Detector::detect_compute(Mat &frame, vector<KeyPoint> &kp, Mat &dp) {
    detector->detect(frame, kp, Mat());
    detector->compute(frame, kp, dp);
}

int Detector::findVehicle(vector<KeyPoint> &kp1, Mat &dp1, vector<KeyPoint> &kp2, Mat &dp2) {
    vector<DMatch> matches;
    int count = 0;
    matcher->match(dp2, dp1, matches);
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance < MAX_DIS) {
            float xDiff = kp2[matches[i].queryIdx].pt.x - kp1[matches[i].trainIdx].pt.x, yDiff = kp2[matches[i].queryIdx].pt.y - kp1[matches[i].trainIdx].pt.y;
            xDiff = xDiff >= 0 ? xDiff : (-1) * xDiff;
            yDiff = yDiff >= 0 ? yDiff : (-1) * yDiff;
            if (!(xDiff > MIN_DIFF && xDiff < MAX_DIFF && yDiff > MIN_DIFF && yDiff < MAX_DIFF)) {
                kp2.erase(kp2.begin() + matches[i].queryIdx);
            } else {
                count++;
            }
        } else {
            kp2.erase(kp2.begin() + matches[i].queryIdx);
        }
    }
    return count;
}

int Detector::matchVehicle(Mat &frame1, vector<KeyPoint> &kp1, Mat &frame2, vector<KeyPoint> &kp2) {
    Mat dp1, dp2;
    vector<DMatch> matches;
    detector->compute(frame1, kp1, dp1);
    detector->compute(frame2, kp2, dp2);
    matcher->match(dp2, dp1, matches);

    int count = 0;
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
    return count;
}



void Detector::detectKeyPoints(Mat start, Mat dest) {
    vector<KeyPoint> keyPoints1;
    Mat descriptor1;

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

void Detector::matchAndDelete() {
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
