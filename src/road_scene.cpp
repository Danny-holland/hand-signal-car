#include "road_scene.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

cv::Scalar bgr(int r, int g, int b) {
    return cv::Scalar(b, g, r);
}

bool blinkOn(float timeSeconds) {
    return std::fmod(timeSeconds, 0.56f) < 0.28f;
}

}  // namespace

RoadScene::RoadScene(int width, int height) : width_(width), height_(height) {}

cv::Point2f RoadScene::carToScreen(cv::Point2f local) const {
    // Local +x is the car's right, local +y is forward (up the road).
    float cosine = std::cos(heading_);
    float sine = std::sin(heading_);
    float screenX = local.x * cosine + local.y * sine;
    float screenY = local.x * sine - local.y * cosine;
    float centerX = width_ * 0.5f + laneOffset_;
    float centerY = height_ * 0.72f;
    return cv::Point2f(centerX + screenX, centerY + screenY);
}

void RoadScene::update(Signal command, float dtSeconds) {
    dtSeconds = std::clamp(dtSeconds, 0.f, 0.05f);
    command_ = command;
    time_ += dtSeconds;

    float targetSpeed = command == Signal::Stop ? 0.f : 260.f;
    float response = command == Signal::Stop ? 3.4f : 1.5f;
    speed_ += (targetSpeed - speed_) * std::min(1.f, response * dtSeconds);
    if (speed_ < 0.5f) {
        speed_ = 0.f;
    }
    scroll_ += speed_ * dtSeconds;

    float steer = 0.f;
    if (command == Signal::LeftTurn) {
        steer = -1.f;
    } else if (command == Signal::RightTurn) {
        steer = 1.f;
    }
    float targetHeading = steer * 0.62f;
    heading_ += (targetHeading - heading_) * std::min(1.f, 5.f * dtSeconds);

    // Steering only changes lanes while the car is actually moving.
    laneOffset_ += std::sin(heading_) * speed_ * dtSeconds * 1.25f;
    float limit = width_ * 0.30f;
    if (laneOffset_ > limit) {
        laneOffset_ = limit;
        if (heading_ > 0.f) {
            heading_ *= 0.85f;
        }
    } else if (laneOffset_ < -limit) {
        laneOffset_ = -limit;
        if (heading_ < 0.f) {
            heading_ *= 0.85f;
        }
    }
}

cv::Mat RoadScene::draw() const {
    cv::Mat panel(height_, width_, CV_8UC3, bgr(118, 158, 86));

    int roadLeft = static_cast<int>(width_ * 0.12f);
    int roadRight = static_cast<int>(width_ * 0.88f);
    cv::rectangle(panel, cv::Rect(roadLeft, 0, roadRight - roadLeft, height_), bgr(58, 60, 64), cv::FILLED);

    cv::line(panel, cv::Point(roadLeft + 8, 0), cv::Point(roadLeft + 8, height_), bgr(230, 230, 230), 4, cv::LINE_AA);
    cv::line(panel, cv::Point(roadRight - 8, 0), cv::Point(roadRight - 8, height_), bgr(230, 230, 230), 4, cv::LINE_AA);

    int lane1 = roadLeft + (roadRight - roadLeft) / 3;
    int lane2 = roadLeft + 2 * (roadRight - roadLeft) / 3;
    const int dash = 36;
    const int gap = 28;
    const int period = dash + gap;
    int offset = static_cast<int>(std::fmod(scroll_, static_cast<float>(period)));
    if (offset < 0) {
        offset += period;
    }
    for (int y = -period + offset; y < height_; y += period) {
        cv::line(panel, cv::Point(lane1, y), cv::Point(lane1, y + dash), bgr(210, 210, 200), 3, cv::LINE_AA);
        cv::line(panel, cv::Point(lane2, y), cv::Point(lane2, y + dash), bgr(210, 210, 200), 3, cv::LINE_AA);
    }

    // Roadside posts make the scrolling obvious when the car is moving.
    const int postPeriod = 150;
    int postOffset = static_cast<int>(std::fmod(scroll_, static_cast<float>(postPeriod)));
    for (int y = -postPeriod + postOffset; y < height_; y += postPeriod) {
        cv::line(panel, cv::Point(roadLeft - 16, y), cv::Point(roadLeft - 16, y + 34), bgr(90, 90, 90), 4, cv::LINE_AA);
        cv::circle(panel, cv::Point(roadLeft - 16, y), 6, bgr(230, 180, 40), cv::FILLED, cv::LINE_AA);
        cv::line(panel, cv::Point(roadRight + 16, y), cv::Point(roadRight + 16, y + 34), bgr(90, 90, 90), 4, cv::LINE_AA);
        cv::circle(panel, cv::Point(roadRight + 16, y), 6, bgr(230, 180, 40), cv::FILLED, cv::LINE_AA);
    }

    cv::Point2f center = carToScreen(cv::Point2f(0.f, 0.f));
    cv::ellipse(panel, cv::Point(static_cast<int>(center.x), static_cast<int>(center.y + 8)),
                cv::Size(28, 46), heading_ * 180.f / 3.1415926f, 0, 360, bgr(40, 70, 40), cv::FILLED, cv::LINE_AA);

    auto poly = [&](const std::vector<cv::Point2f>& local, const cv::Scalar& color) {
        std::vector<cv::Point> screen;
        screen.reserve(local.size());
        for (cv::Point2f point : local) {
            cv::Point2f mapped = carToScreen(point);
            screen.emplace_back(cv::Point(cvRound(mapped.x), cvRound(mapped.y)));
        }
        cv::fillConvexPoly(panel, screen, color, cv::LINE_AA);
    };

    auto wheel = [&](cv::Point2f local) {
        poly({local + cv::Point2f(-7.f, -14.f), local + cv::Point2f(7.f, -14.f),
              local + cv::Point2f(7.f, 14.f), local + cv::Point2f(-7.f, 14.f)},
             bgr(28, 28, 30));
    };
    wheel(cv::Point2f(-24.f, 30.f));
    wheel(cv::Point2f(24.f, 30.f));
    wheel(cv::Point2f(-24.f, -30.f));
    wheel(cv::Point2f(24.f, -30.f));

    poly({{-20.f, -46.f}, {20.f, -46.f}, {22.f, 34.f}, {-22.f, 34.f}}, bgr(40, 110, 205));
    poly({{-15.f, -6.f}, {15.f, -6.f}, {13.f, 28.f}, {-13.f, 28.f}}, bgr(25, 70, 150));
    poly({{-12.f, 8.f}, {12.f, 8.f}, {10.f, 24.f}, {-10.f, 24.f}}, bgr(190, 210, 220));

    bool braking = command_ == Signal::Stop;
    cv::Scalar brake = braking ? bgr(40, 40, 230) : bgr(50, 50, 120);
    poly({{-16.f, -46.f}, {-6.f, -46.f}, {-6.f, -40.f}, {-16.f, -40.f}}, brake);
    poly({{6.f, -46.f}, {16.f, -46.f}, {16.f, -40.f}, {6.f, -40.f}}, brake);

    bool lights = blinkOn(time_);
    cv::Scalar leftLamp = (command_ == Signal::LeftTurn && lights) ? bgr(40, 190, 255) : bgr(40, 90, 120);
    cv::Scalar rightLamp = (command_ == Signal::RightTurn && lights) ? bgr(40, 190, 255) : bgr(40, 90, 120);
    poly({{-22.f, 28.f}, {-14.f, 28.f}, {-14.f, 36.f}, {-22.f, 36.f}}, leftLamp);
    poly({{14.f, 28.f}, {22.f, 28.f}, {22.f, 36.f}, {14.f, 36.f}}, rightLamp);

    std::string motion = stopped() ? "stopped" : (speed_ < 180.f ? "slowing" : "driving");
    cv::putText(panel, motion, cv::Point(16, height_ - 18), cv::FONT_HERSHEY_SIMPLEX, 0.6, bgr(245, 245, 245), 2,
                cv::LINE_AA);
    return panel;
}
