# OpenCV_detector

This is a project on vehicle detection with ORB method.

Ver 1.0
------------------
1. Basic feature points detection, sift out invalid distance

To do: calculate vehicle numbers, minimize measurement error.

Ver 1.1
------------------
1. Improvement: detect the moving cars(through cooresponding feature points movement)
   With this method, most of the measurement errors can be eliminated. However, some bicyles and motors may
   also be detected.

To do: adjust the MIN_DIF and MAX_DIF to minimize the error, count a cluster of points as a single car

Ver 1.2
------------------
1. circle the adjacent feature points as a single car, may exist measurement error
