#pragma once

#include "signals.hpp"

#include <opencv2/core.hpp>

// Top-down road. The car drives toward the top of the panel.
// Left and right signals steer, and stop slows the car to a halt.
class RoadScene {
public:
    RoadScene(int width, int height);

    void update(Signal command, float dtSeconds);
    cv::Mat draw() const;

    float speed() const { return speed_; }
    bool stopped() const { return speed_ < 18.f; }

private:
    cv::Point2f carToScreen(cv::Point2f local) const;

    int width_;
    int height_;
    float speed_ = 0.f;
    float scroll_ = 0.f;
    float laneOffset_ = 0.f;
    float heading_ = 0.f;
    float time_ = 0.f;
    Signal command_ = Signal::Straight;
};
