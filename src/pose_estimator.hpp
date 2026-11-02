#pragma once

#include "signals.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <array>
#include <string>

struct Keypoint {
    cv::Point2f point;
    float score = 0.f;
};

// COCO body order used by YOLOv8-pose: 5/7/9 are the left shoulder, elbow, and wrist.
struct Pose {
    std::array<Keypoint, 17> points{};
    bool found = false;
    float score = 0.f;
};

class PoseEstimator {
public:
    bool load(const std::string& onnxPath);
    bool loaded() const { return ready_; }
    const std::string& error() const { return error_; }

    // BGR image. Returns the highest-confidence person, with keypoints in image pixels.
    Pose detect(const cv::Mat& bgr);

private:
    cv::dnn::Net net_;
    bool ready_ = false;
    std::string error_;
    int inputSize_ = 640;
};

Signal classifyLeftArm(const Pose& pose);
