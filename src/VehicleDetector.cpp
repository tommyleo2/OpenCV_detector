#include "VehicleDetector.h"

VehicleDetector::VehicleDetector(bool _start) {
    start = _start;
    if (!start) {
        KP.push_back(vector<KeyPoint>());
        DP.push_back(Mat());
    }
    detector = ORB::create();
    matcher = DescriptorMatcher::create("BruteForce-Hamming");
}

int VehicleDetector::findVehicle(Mat &frame1, Mat &frame2) {
    /*
      This algorithm compares two frames taken in sequence in a very short time.
      Match the points which also have small distance from the two frame.
      And those points can be later considered as keypoints of cars.
    */
    vector<KeyPoint> _kp1, _kp2;
    Mat _dp1, _dp2;
    vector<DMatch> matches;
    int count = 0;
    detector->detect(frame1, _kp1, Mat());
    detector->compute(frame1, _kp1, _dp1);
    detector->detect(frame2, _kp2, Mat());
    detector->compute(frame2, _kp2, _dp2);
    matcher->match(_dp2, _dp1, matches);
    priority_queue<int> deleteQueue;
    for (int i = 0; i < matches.size(); i++) {
        if (1/*matches[i].distance < MAX_DIS*/) {
            float xDiff = _kp2[matches[i].queryIdx].pt.x - _kp1[matches[i].trainIdx].pt.x,
                  yDiff = _kp2[matches[i].queryIdx].pt.y - _kp1[matches[i].trainIdx].pt.y;
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
        _kp2.erase(_kp2.begin() + deleteQueue.top());
        deleteQueue.pop();
    }
    detector->compute(frame2, _kp2, _dp2);
    if (!start) {
        KP.pop_back();
        DP.pop_back();
    }
    KP.push_back(_kp2);
    DP.push_back(_dp2);
    return count;
}
