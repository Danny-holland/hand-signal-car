#include "pose_estimator.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

constexpr int kLeftShoulder = 5;
constexpr int kRightShoulder = 6;
constexpr int kLeftElbow = 7;
constexpr int kLeftWrist = 9;

float length(cv::Point2f v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

float angleAtElbow(cv::Point2f shoulder, cv::Point2f elbow, cv::Point2f wrist) {
    cv::Point2f toShoulder = shoulder - elbow;
    cv::Point2f toWrist = wrist - elbow;
    float denom = length(toShoulder) * length(toWrist);
    if (denom < 1.f) {
        return 0.f;
    }
    float cosine = std::clamp((toShoulder.x * toWrist.x + toShoulder.y * toWrist.y) / denom, -1.f, 1.f);
    return std::acos(cosine);
}

cv::Mat letterbox(const cv::Mat& src, int size, float& scale, int& padX, int& padY) {
    scale = std::min(size / static_cast<float>(src.cols), size / static_cast<float>(src.rows));
    int resizedW = std::max(1, static_cast<int>(std::round(src.cols * scale)));
    int resizedH = std::max(1, static_cast<int>(std::round(src.rows * scale)));
    padX = (size - resizedW) / 2;
    padY = (size - resizedH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resizedW, resizedH));
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padX, padY, resizedW, resizedH)));
    return canvas;
}

}  // namespace

bool PoseEstimator::load(const std::string& onnxPath) {
    ready_ = false;
    error_.clear();
    try {
        net_ = cv::dnn::readNetFromONNX(onnxPath);
        if (net_.empty()) {
            error_ = "OpenCV could not read the pose model.";
            return false;
        }
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        ready_ = true;
        return true;
    } catch (const cv::Exception& ex) {
        error_ = ex.what();
        return false;
    }
}

Pose PoseEstimator::detect(const cv::Mat& bgr) {
    Pose pose;
    if (!ready_ || bgr.empty()) {
        return pose;
    }

    float scale = 1.f;
    int padX = 0;
    int padY = 0;
    cv::Mat input = letterbox(bgr, inputSize_, scale, padX, padY);
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(inputSize_, inputSize_),
                                          cv::Scalar(), true, false);
    net_.setInput(blob);
    cv::Mat out = net_.forward();
    if (out.empty()) {
        return pose;
    }

    static bool loggedShape = false;
    if (!loggedShape) {
        loggedShape = true;
        std::cout << "Pose model output shape:";
        for (int i = 0; i < out.dims; ++i) {
            std::cout << (i == 0 ? " " : "x") << out.size[i];
        }
        std::cout << "\n";
    }

    // Ultralytics exports either [1, 56, 8400] or [1, 8400, 56]. Each anchor is
    // box (4), person score (1), then 17 keypoints as x, y, score.
    cv::Mat pred(out.size[out.dims - 2], out.size[out.dims - 1], CV_32F, out.ptr<float>());
    if (pred.cols > pred.rows) {
        pred = pred.t();
    }
    if (!pred.isContinuous()) {
        pred = pred.clone();
    }
    if (pred.cols < 56) {
        return pose;
    }

    int best = -1;
    float bestScore = 0.35f;
    for (int i = 0; i < pred.rows; ++i) {
        float score = pred.at<float>(i, 4);
        if (score > bestScore) {
            bestScore = score;
            best = i;
        }
    }
    if (best < 0) {
        return pose;
    }

    const float* row = pred.ptr<float>(best);
    bool normalized = row[0] <= 2.f && row[1] <= 2.f;
    pose.found = true;
    pose.score = bestScore;
    for (int k = 0; k < 17; ++k) {
        float x = row[5 + k * 3];
        float y = row[6 + k * 3];
        float kpScore = row[7 + k * 3];
        if (normalized) {
            x *= static_cast<float>(inputSize_);
            y *= static_cast<float>(inputSize_);
        }
        pose.points[k].point = cv::Point2f((x - static_cast<float>(padX)) / scale,
                                           (y - static_cast<float>(padY)) / scale);
        pose.points[k].score = kpScore;
    }
    return pose;
}

Signal classifyLeftArm(const Pose& pose) {
    if (!pose.found) {
        return Signal::Unknown;
    }

    const Keypoint& shoulder = pose.points[kLeftShoulder];
    const Keypoint& otherShoulder = pose.points[kRightShoulder];
    const Keypoint& elbow = pose.points[kLeftElbow];
    const Keypoint& wrist = pose.points[kLeftWrist];
    if (shoulder.score < 0.30f || elbow.score < 0.30f || wrist.score < 0.22f) {
        return Signal::Unknown;
    }

    // "Out" is away from the chest, not toward the left edge of the screen.
    // A mirrored webcam puts the left arm on either side of the image.
    float shoulderWidth = 80.f;
    float outSign = -1.f;
    if (otherShoulder.score >= 0.30f) {
        shoulderWidth = std::max(30.f, length(otherShoulder.point - shoulder.point));
        outSign = shoulder.point.x <= otherShoulder.point.x ? -1.f : 1.f;
    }
    auto outward = [&](cv::Point2f point) { return (point.x - shoulder.point.x) * outSign; };

    cv::Point2f forearm = wrist.point - elbow.point;
    float upperLen = length(elbow.point - shoulder.point);
    float forearmLen = length(forearm);
    if (upperLen < shoulderWidth * 0.20f || forearmLen < shoulderWidth * 0.20f) {
        return Signal::Unknown;
    }

    float elbowAngle = angleAtElbow(shoulder.point, elbow.point, wrist.point);
    float wristOut = outward(wrist.point);
    float elbowOut = outward(elbow.point);
    float drop = wrist.point.y - shoulder.point.y;

    bool forearmUp = forearm.y < -std::max(forearmLen * 0.45f, shoulderWidth * 0.25f);
    bool handAboveShoulder = wrist.point.y < shoulder.point.y - shoulderWidth * 0.28f;
    bool handAboveElbow = wrist.point.y < elbow.point.y - shoulderWidth * 0.12f;
    if (forearmUp && handAboveShoulder && handAboveElbow && elbowAngle < 2.9f) {
        return Signal::RightTurn;
    }

    // Stop: elbow held out from the body and the forearm hanging down.
    bool forearmDown = forearm.y > std::max(forearmLen * 0.45f, shoulderWidth * 0.25f) &&
                       forearm.y > std::abs(forearm.x) * 0.75f;
    bool handBelowElbow = wrist.point.y > elbow.point.y + shoulderWidth * 0.20f;
    bool elbowLateral = elbowOut > shoulderWidth * 0.18f;
    float elbowDrop = std::max(0.f, elbow.point.y - shoulder.point.y);
    bool elbowNotHanging = elbowOut > elbowDrop * 0.40f;
    if (forearmDown && handBelowElbow && elbowLateral && elbowNotHanging && elbowAngle < 2.95f) {
        return Signal::Stop;
    }

    // Left: the whole arm reaches out sideways. A small droop is still a left turn.
    bool extended = wristOut > shoulderWidth * 0.75f;
    bool mostlyHorizontal = wristOut > std::abs(drop) * 0.70f;
    bool longEnough = length(wrist.point - shoulder.point) > shoulderWidth * 0.95f;
    bool handNotHigh = drop > -shoulderWidth * 0.25f;
    if (extended && mostlyHorizontal && longEnough && handNotHigh) {
        return Signal::LeftTurn;
    }
    return Signal::Straight;
}
