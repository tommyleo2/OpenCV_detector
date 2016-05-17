#include "DetectorManager.h"

DetectorManager *DetectorManager::detectorManager = NULL;
int DetectorManager::maxPoolSize = 0;
bool DetectorManager::_start = false;
std::function<void (int, int, int)> DetectorManager::displayResult = std::function<void (int, int, int)>();
DebugFlag::DEBUG_FLAG DetectorManager::debug = DebugFlag::NO_DEBUG_INFO;

DetectorManager::DetectorManager() {
    _start = false;
    _pause = false;
}

DetectorManager *DetectorManager::getInstance() {
    if (detectorManager == NULL && maxPoolSize != 0) {
        detectorManager = new DetectorManager();
        return detectorManager;
    }
    return NULL;
}

void DetectorManager::setDetectorManager(int _fps,
                                         std::function<void (int, int, int)> _dr,
                                         DebugFlag::DEBUG_FLAG _debug) {
    if (_start) {
        return;
    }
    maxPoolSize = _fps * RETAIN_TIME;
    displayResult = _dr;
    debug = _debug;
}

void DetectorManager::pushFrame(TwoMat &frame) {
    rawFramePool.push(frame);
}

void DetectorManager::addStart(Range X, Range Y, int tag) {
    if (_start) {
        return;
    }
    startX.push_back(X);
    startY.push_back(Y);
    startTags.push_back(tag);
}

void DetectorManager::addDest(Range X, Range Y, int tag) {
    if (_start) {
        return;
    }
    destX.push_back(X);
    destY.push_back(Y);
    destTags.push_back(tag);
}

bool DetectorManager::start() {
    if (_start) {
        return false;
    }
    _start = true;
    createDetectors();

    mainThread = thread(loopDetectWrapper);

    return true;
}

void DetectorManager::terminate() {
    _start = false;
}

void DetectorManager::pause() {
    _pause = true;
}

void DetectorManager::resume() {
    _pause = false;
}

void DetectorManager::loopDetectWrapper() {
    detectorManager->loopDetect();
}

void DetectorManager::findStartVehicleWrapper(int num, Mat start, Mat dest) {
    detectorManager->startDetectors[num].findVehicle(start, dest);
}

void DetectorManager::findDestVehicleWrapper(int num, Mat start, Mat dest) {
    detectorManager->destDetectors[num].findVehicle(start, dest);
}

void DetectorManager::vehicleMatcherWrapper(int dest, int start) {
    detectorManager->vehicleMatcher(dest, start);
}

void DetectorManager::loopDetect() {
    while (_start) {
        detect();
        match();
        deleteOutdatedInfo();
        while (_pause);
    }
}

void DetectorManager::createDetectors() {
    for (int i = 0; i < startX.size(); ++i) {
        startDetectors.push_back(VehicleDetector(true));
    }
    for (int i = 0; i < destX.size(); ++i) {
        destDetectors.push_back(VehicleDetector(false));
    }
    for (int i = 0; i < destX.size() * startX.size(); ++i) {
            matchers.push_back(DescriptorMatcher::create("BruteForce-Hamming"));
    }
}

void DetectorManager::detect() {
    vector<thread> startTh, destTh;
    while (rawFramePool.size() < 3);
    TwoMat raw = rawFramePool.front();
    rawFramePool.pop();
    for (int i = 0; i < startDetectors.size(); i++) {
        startTh.push_back(thread(findStartVehicleWrapper,
                                 i,
                                 raw.first(startX[i], startY[i]), raw.second(startX[i], startY[i])));
    }
    for (int i = 0; i < destDetectors.size(); ++i) {
        destTh.push_back(thread(findDestVehicleWrapper,
                                i,
                                raw.first(destX[i], destY[i]), raw.second(destX[i], destY[i])));
    }
    for (int i = 0; i < startTh.size(); ++i) {
        startTh[i].join();
    }
    for (int i = 0; i < destTh.size(); ++i) {
        destTh[i].join();
    }
    startTh.clear();
    destTh.clear();
}

void DetectorManager::match() {
    vector<thread> matchers;
    for (int i = 0; i < destDetectors.size(); ++i) {
        for (int j = 0; j < startDetectors.size(); ++j) {
            matchers.push_back(thread(vehicleMatcherWrapper, i, j));
        }
    }
    for (int i = 0; i < matchers.size(); ++i) {
        matchers[i].join();
    }
}

void DetectorManager::deleteOutdatedInfo() {
    while (startDetectors[0].KP.size() > maxPoolSize) {
        for (int i = 0; i < startDetectors.size(); ++i) {
            startDetectors[i].KP.pop_front();
        }
    }
}

void DetectorManager::vehicleMatcher(int dest, int start) {
    vector<KeyPoint> &kp2 = destDetectors[dest].KP.front();
    Mat &dp2 = destDetectors[dest].DP.front();
    int count = 0;
    auto itKP = startDetectors[start].KP.begin();
    auto itDP = startDetectors[start].DP.begin();
    for (; itKP != startDetectors[start].KP.end(); ++itKP, ++itDP) {
        vector<KeyPoint> &kp1 = *itKP;
        Mat &dp1 = *itDP;
        vector<DMatch> matches;
        Ptr<DescriptorMatcher> matcher = matchers[dest * destDetectors.size() + start];
        //  set destination area as the query area and start area as the train area
        //  this makes sure that every car in destination area will be matched (if match succeed)
        matcher->match(dp2, dp1, matches);
        //  mark the point if it has already been in a circle
        bool *mark = new bool[matches.size()];
        for (int i = 0; i < matches.size(); i++) {
            mark[i] = false;
        }

        for (int i = 0; i < matches.size(); i++) {
            if (mark[i]) {
                continue;
            }
            //  start a new circle
            mark[i] = true;
            vector<DMatch> cluster;
            cluster.push_back(matches[i]);
            //  BFS algorithm
            for (int j = 1 + i; j < matches.size(); j++) {
                if (mark[j]) {
                    continue;
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
                    //  judge if the point is in the current circle
                    if (sxdiff < MAX_RANGE && sydiff < MAX_RANGE && dxdiff < MAX_RANGE && dydiff < MAX_RANGE) {
                        mark[j] = true;
                        cluster.push_back(matches[j]);
                        j = i;
                        break;
                    }
                }
            }
            if (cluster.size() > MIN_CLUSTER) {
                count++;
            }
        }
        delete[] mark;
        displayResult(startTags[start], destTags[dest], count);
    }
}
