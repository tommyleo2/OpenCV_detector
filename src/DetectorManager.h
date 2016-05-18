#ifndef DETECTORMANAGER_H
#define DETECTORMANAGER_H

/*
  Usage:
  1. Set the DetectorManager.
  2. Get an instance of DetectorManager.
     If you apply for more than once, it will return NULL except for the first time.
     If you haven't set it yet, it will also return NULL.'
  3. Add the all the areas that you want to detect.
  4. Start detection. Once you get started, you cannot set the manager and add area again.

  Note:
  1. Tags are used to represent the detecting area and they will be only used on the display function.
*/


#include <opencv2/opencv.hpp>
#include <iterator>
#include <vector>
#include <list>
#include <queue>
#include <thread>
#include <functional>
#include "Queue.h"
#include "VehicleDetector.h"

#define CARDETECTED 20
#define MAXDIFF 10
#define MAX_RANGE 15 //maximum distance of keypoints in the same vehicle
#define MIN_CLUSTER 5 //minimun number of keypoints to define a car
#define RETAIN_TIME 10 //each frame will be retained for 10s in pool

namespace DebugFlag {
    enum DEBUG_FLAG {
        NO_DEBUG_INFO,
        SHOW_MATCH_RESULT,
        SHOW_STEP
    };
}

using std::vector;
using namespace cv;


typedef struct TwoMat {
    TwoMat() {}
    TwoMat(Mat _first, Mat _second) {
        first = _first.clone();
        second = _second.clone();
    }
    TwoMat(const TwoMat &_twoMat) {
        first = _twoMat.first.clone();
        second = _twoMat.second.clone();
    }
    const TwoMat &operator=(const TwoMat &_twoMat) {
        first = _twoMat.first.clone();
        second = _twoMat.second.clone();
        return *this;
    }
    Mat first, second;
} TwoMat;

class DetectorManager {
public:
    static DetectorManager *getInstance();
    static void setDetectorManager(int _fps,
                                   std::function<void (int, int, int)> _dr,
                                   DebugFlag::DEBUG_FLAG _debug = DebugFlag::NO_DEBUG_INFO);
    void pushFrame(TwoMat &frame);
    //  add detecting area, use tags to specify areas
    //  if detecting job has already started, adding detecting area cannot be done
    void addStart(Range X, Range Y, int tag);
    void addDest(Range X, Range Y, int tag);
    //  start all detectors, return false if no detector exists
    bool start();
    //  terminate the detecting job
    void terminate();
    //  pause the job
    void pause();
    //  resume the job
    void resume();

private:
    //Singleton, disable copy and assign
    DetectorManager();
    DetectorManager(const DetectorManager &) {}
    DetectorManager &operator=(const DetectorManager &) {}
    //  not available in multithreading
    static DetectorManager *detectorManager;

    static int maxPoolSize;
    static DebugFlag::DEBUG_FLAG debug;
    static std::function<void (int startTag, int destTag, int carNum)> displayResult;
    //  if detectors have already been working, detectorManager will be not able be modify
    static bool _start;

    //  adaptor for making functions mutithread
    static void loopDetectWrapper();
    static void findStartVehicleWrapper(int num, Mat start, Mat dest);
    static void findDestVehicleWrapper(int num, Mat start, Mat dest);
    static void vehicleMatcherWrapper(int dest, int start);

    thread mainThread;

    bool _pause;

    void loopDetect();
    void createDetectors();
    void detect();
    void match();
    void deleteOutdatedInfo();
    void vehicleMatcher(int dest, int start);

    Queue<TwoMat> rawFramePool;
    vector<Range> startX, startY, destX, destY;
    vector<int> startTags, destTags;
    vector<VehicleDetector> startDetectors, destDetectors;

    vector< Ptr<DescriptorMatcher> > matchers;
};

#endif /* DETECTORMANAGER_H */
