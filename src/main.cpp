#include "pose_estimator.hpp"
#include "road_scene.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr int kRoadW = 460;
constexpr int kRoadH = 640;
constexpr int kCamW = 780;
constexpr int kGuideH = 168;
constexpr int kStatusH = 78;

const cv::Scalar kLeftShoulderColor(0, 220, 255);
const int kSkeleton[][2] = {
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12},
    {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, {0, 5}, {0, 6},
};

std::filesystem::path findFile(const std::filesystem::path& argv0, const std::filesystem::path& relative) {
    std::vector<std::filesystem::path> roots = {
        std::filesystem::current_path(),
        std::filesystem::absolute(argv0).parent_path(),
        std::filesystem::absolute(argv0).parent_path().parent_path().parent_path(),
    };
    for (const auto& root : roots) {
        std::filesystem::path candidate = root / relative;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

cv::Scalar signalColor(Signal signal) {
    switch (signal) {
    case Signal::LeftTurn: return cv::Scalar(40, 160, 40);
    case Signal::RightTurn: return cv::Scalar(180, 120, 20);
    case Signal::Stop: return cv::Scalar(40, 40, 200);
    case Signal::Unknown: return cv::Scalar(80, 80, 80);
    case Signal::Straight: return cv::Scalar(60, 60, 60);
    }
    return cv::Scalar(60, 60, 60);
}

void blitFit(cv::Mat& dest, cv::Rect area, const cv::Mat& source, const cv::Scalar& background) {
    dest(area).setTo(background);
    if (source.empty() || area.width <= 0 || area.height <= 0) {
        return;
    }
    double scale = std::min(area.width / static_cast<double>(source.cols),
                             area.height / static_cast<double>(source.rows));
    int width = std::max(1, static_cast<int>(std::round(source.cols * scale)));
    int height = std::max(1, static_cast<int>(std::round(source.rows * scale)));
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(width, height));
    int x = area.x + (area.width - width) / 2;
    int y = area.y + (area.height - height) / 2;
    resized.copyTo(dest(cv::Rect(x, y, width, height)));
}

void drawPose(cv::Mat& frame, const Pose& pose) {
    if (!pose.found) {
        return;
    }
    for (const auto& pair : kSkeleton) {
        const Keypoint& a = pose.points[pair[0]];
        const Keypoint& b = pose.points[pair[1]];
        if (a.score < 0.3f || b.score < 0.3f) {
            continue;
        }
        bool leftArm = (pair[0] == 5 && pair[1] == 7) || (pair[0] == 7 && pair[1] == 9);
        cv::line(frame, a.point, b.point, leftArm ? kLeftShoulderColor : cv::Scalar(255, 180, 60), leftArm ? 6 : 3,
                 cv::LINE_AA);
    }
    for (int i = 0; i < 17; ++i) {
        if (pose.points[i].score < 0.3f) {
            continue;
        }
        bool leftArm = i == 5 || i == 7 || i == 9;
        cv::circle(frame, pose.points[i].point, leftArm ? 7 : 4, leftArm ? kLeftShoulderColor : cv::Scalar(255, 255, 255),
                   cv::FILLED, cv::LINE_AA);
    }
}

class SignalSmoother {
public:
    Signal update(Signal raw) {
        if (raw == Signal::Unknown) {
            ++unknownFrames_;
            if (unknownFrames_ > 18) {
                stable_ = Signal::Straight;
                pendingFrames_ = 0;
            }
            return stable_;
        }
        unknownFrames_ = 0;
        if (raw == stable_) {
            pendingFrames_ = 0;
            return stable_;
        }
        if (raw != pending_) {
            pending_ = raw;
            pendingFrames_ = 1;
        } else {
            ++pendingFrames_;
        }
        int needed = raw == Signal::Straight ? 5 : 3;
        if (pendingFrames_ >= needed) {
            stable_ = raw;
        }
        return stable_;
    }

private:
    Signal stable_ = Signal::Straight;
    Signal pending_ = Signal::Straight;
    int pendingFrames_ = 0;
    int unknownFrames_ = 0;
};

class PoseSmoother {
public:
    Pose apply(const Pose& current) {
        if (!current.found) {
            has_ = false;
            return current;
        }
        if (!has_) {
            previous_ = current;
            has_ = true;
            return current;
        }
        Pose blended = current;
        for (int i = 0; i < 17; ++i) {
            if (current.points[i].score > 0.2f && previous_.points[i].score > 0.2f) {
                blended.points[i].point = previous_.points[i].point * 0.55f + current.points[i].point * 0.45f;
            }
        }
        previous_ = blended;
        return blended;
    }

private:
    Pose previous_{};
    bool has_ = false;
};

bool rightArmIsSignaling(const Pose& pose) {
    if (!pose.found) {
        return false;
    }
    const Keypoint& shoulder = pose.points[6];
    const Keypoint& elbow = pose.points[8];
    const Keypoint& wrist = pose.points[10];
    const Keypoint& leftWrist = pose.points[9];
    const Keypoint& leftShoulder = pose.points[5];
    if (shoulder.score < 0.35f || wrist.score < 0.35f || elbow.score < 0.35f) {
        return false;
    }
    float rightReach = static_cast<float>(cv::norm(wrist.point - shoulder.point));
    float leftReach = (leftWrist.score > 0.3f && leftShoulder.score > 0.3f)
                          ? static_cast<float>(cv::norm(leftWrist.point - leftShoulder.point))
                          : 0.f;
    float shoulderWidth = 80.f;
    if (leftShoulder.score > 0.3f) {
        shoulderWidth = std::max(20.f, static_cast<float>(cv::norm(shoulder.point - leftShoulder.point)));
    }
    return rightReach > shoulderWidth * 1.3f && rightReach > leftReach * 1.25f;
}

cv::VideoCapture openCamera() {
    for (int index = 0; index < 3; ++index) {
        cv::VideoCapture capture(index, cv::CAP_DSHOW);
        if (capture.isOpened()) {
            capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
            std::cout << "Camera opened on device " << index << "\n";
            return capture;
        }
    }
    return {};
}

void putLabel(cv::Mat& image, const std::string& text, cv::Point origin, double scale, const cv::Scalar& color,
              int thickness = 2) {
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

Pose syntheticPose(cv::Point2f shoulder, cv::Point2f elbow, cv::Point2f wrist, float otherShoulderX = -1.f) {
    Pose pose;
    pose.found = true;
    pose.score = 1.f;
    pose.points[5] = {shoulder, 1.f};
    float otherX = otherShoulderX < 0.f ? shoulder.x + 80.f : otherShoulderX;
    pose.points[6] = {cv::Point2f(otherX, shoulder.y), 1.f};
    pose.points[7] = {elbow, 1.f};
    pose.points[9] = {wrist, 1.f};
    return pose;
}

int runSelfTest(const std::filesystem::path& modelPath) {
    struct Expectation {
        const char* name;
        Pose pose;
        Signal expected;
    };
    Expectation cases[] = {
        {"left turn", syntheticPose({200.f, 200.f}, {120.f, 200.f}, {40.f, 205.f}), Signal::LeftTurn},
        {"left turn, arm droops", syntheticPose({200.f, 200.f}, {130.f, 250.f}, {40.f, 290.f}), Signal::LeftTurn},
        {"left turn, other image side", syntheticPose({200.f, 200.f}, {280.f, 205.f}, {360.f, 210.f}, 120.f), Signal::LeftTurn},
        {"right turn", syntheticPose({200.f, 200.f}, {130.f, 210.f}, {120.f, 70.f}), Signal::RightTurn},
        {"right turn, high elbow", syntheticPose({200.f, 200.f}, {150.f, 140.f}, {145.f, 40.f}), Signal::RightTurn},
        {"stop", syntheticPose({200.f, 200.f}, {120.f, 215.f}, {110.f, 330.f}), Signal::Stop},
        {"stop, elbow closer", syntheticPose({200.f, 200.f}, {155.f, 245.f}, {148.f, 345.f}), Signal::Stop},
        {"stop, other image side", syntheticPose({200.f, 200.f}, {275.f, 230.f}, {285.f, 340.f}, 120.f), Signal::Stop},
        {"arm down", syntheticPose({200.f, 200.f}, {190.f, 310.f}, {185.f, 400.f}), Signal::Straight},
    };

    int failures = 0;
    for (const Expectation& test : cases) {
        Signal got = classifyLeftArm(test.pose);
        std::cout << test.name << ": " << signalLabel(got) << "\n";
        if (got != test.expected) {
            std::cout << "  expected " << signalLabel(test.expected) << "\n";
            ++failures;
        }
    }

    PoseEstimator estimator;
    if (!estimator.load(modelPath.string())) {
        std::cout << "Model failed to load: " << estimator.error() << "\n";
        return 1;
    }
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(32, 32, 32));
    cv::circle(blank, cv::Point(320, 180), 40, cv::Scalar(180, 180, 180), cv::FILLED);
    estimator.detect(blank);

    cv::Mat sheet(kRoadH, kRoadW * 3 + 20, CV_8UC3, cv::Scalar(20, 20, 20));
    Signal commands[] = {Signal::LeftTurn, Signal::RightTurn, Signal::Stop};
    for (int i = 0; i < 3; ++i) {
        RoadScene scene(kRoadW, kRoadH);
        for (int step = 0; step < 50; ++step) {
            scene.update(commands[i], 0.05f);
        }
        scene.draw().copyTo(sheet(cv::Rect(i * (kRoadW + 10), 0, kRoadW, kRoadH)));
    }
    cv::imwrite("self_test_output.png", sheet);
    std::cout << (failures == 0 ? "Self-test passed\n" : "Self-test failed\n");
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path argv0 = argc > 0 ? argv[0] : "hand_signal_car";
    std::filesystem::path modelPath = findFile(argv0, "models/yolov8n-pose.onnx");
    std::filesystem::path guidePath = findFile(argv0, "hand_signals.jpg");
    if (guidePath.empty()) {
        guidePath = findFile(argv0, "top-hand-signals-driving-1166-image.jpg");
    }

    bool selfTest = argc > 1 && std::string(argv[1]) == "--self-test";
    if (selfTest) {
        if (modelPath.empty()) {
            std::cout << "Missing models/yolov8n-pose.onnx\n";
            return 1;
        }
        return runSelfTest(modelPath);
    }

    PoseEstimator estimator;
    if (modelPath.empty()) {
        std::cout << "Could not find models/yolov8n-pose.onnx\n";
        std::cout << "Run scripts/setup.ps1, then scripts/build.ps1.\n";
    } else if (!estimator.load(modelPath.string())) {
        std::cout << "Could not load the pose model:\n" << estimator.error() << "\n";
    } else {
        std::cout << "Loaded pose model " << modelPath << "\n";
    }

    cv::Mat guide = guidePath.empty() ? cv::Mat() : cv::imread(guidePath.string());
    cv::VideoCapture camera = openCamera();
    if (!camera.isOpened()) {
        std::cout << "No camera found. Press A/D/S/W to drive the car manually.\n";
    }

    RoadScene road(kRoadW, kRoadH);
    SignalSmoother smoother;
    PoseSmoother poseSmoother;
    std::optional<Signal> manual;
    auto previousTime = std::chrono::steady_clock::now();
    float fps = 0.f;

    const int canvasW = kCamW + kRoadW;
    const int canvasH = kRoadH + kGuideH + kStatusH;
    const std::string window = "Hand Signal Car";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);

    while (true) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - previousTime).count();
        previousTime = now;
        fps = fps * 0.9f + (1.f / std::max(dt, 0.001f)) * 0.1f;

        cv::Mat frame;
        bool gotFrame = camera.isOpened() && camera.read(frame) && !frame.empty();
        if (gotFrame) {
            cv::flip(frame, frame, 1);
        }

        Pose pose;
        if (gotFrame && estimator.loaded()) {
            pose = poseSmoother.apply(estimator.detect(frame));
            drawPose(frame, pose);
        }
        Signal seen = pose.found ? classifyLeftArm(pose) : Signal::Unknown;
        Signal stable = smoother.update(seen);
        Signal command = manual.value_or(stable);

        road.update(command, dt);

        cv::Mat canvas(canvasH, canvasW, CV_8UC3, cv::Scalar(236, 232, 226));
        cv::Mat cameraPanel;
        if (gotFrame) {
            cameraPanel = frame;
        } else {
            cameraPanel = cv::Mat(480, 640, CV_8UC3, cv::Scalar(48, 48, 48));
            putLabel(cameraPanel, "No camera", cv::Point(180, 230), 1.2, cv::Scalar(240, 240, 240), 2);
            putLabel(cameraPanel, "A left   D right   S stop   W go", cv::Point(70, 280), 0.7, cv::Scalar(200, 200, 200), 1);
        }
        blitFit(canvas, cv::Rect(0, 0, kCamW, kRoadH), cameraPanel, cv::Scalar(32, 32, 32));

        std::string seenText = std::string("Seen: ") + signalLabel(seen);
        putLabel(canvas, seenText, cv::Point(24, 42), 0.9, cv::Scalar(255, 255, 255), 2);
        if (manual) {
            putLabel(canvas, "MANUAL", cv::Point(24, 84), 0.7, cv::Scalar(80, 220, 255), 2);
        } else if (!estimator.loaded()) {
            putLabel(canvas, "Pose model not loaded", cv::Point(24, 84), 0.7, cv::Scalar(80, 80, 255), 2);
        } else if (!pose.found) {
            putLabel(canvas, "Step into frame", cv::Point(24, 84), 0.7, cv::Scalar(220, 220, 220), 2);
        } else if (seen == Signal::Unknown && rightArmIsSignaling(pose)) {
            putLabel(canvas, "Use your left arm", cv::Point(24, 84), 0.7, cv::Scalar(80, 220, 255), 2);
        }
        putLabel(canvas, "FPS " + std::to_string(static_cast<int>(fps + 0.5f)), cv::Point(kCamW - 130, 42), 0.6,
                 cv::Scalar(230, 230, 230), 1);

        road.draw().copyTo(canvas(cv::Rect(kCamW, 0, kRoadW, kRoadH)));
        blitFit(canvas, cv::Rect(0, kRoadH, canvasW, kGuideH), guide, cv::Scalar(232, 240, 244));
        if (guide.empty()) {
            putLabel(canvas, "Left arm straight out = left    forearm up = right    forearm down = stop",
                     cv::Point(40, kRoadH + 90), 0.7, cv::Scalar(40, 40, 40), 2);
        }

        cv::Rect status(0, kRoadH + kGuideH, canvasW, kStatusH);
        canvas(status).setTo(signalColor(command));
        putLabel(canvas, signalLabel(command), cv::Point(24, status.y + 52), 1.3, cv::Scalar(255, 255, 255), 2);
        putLabel(canvas, "V camera    A/D turn    S stop    W straight    Q quit", cv::Point(330, status.y + 48), 0.55,
                 cv::Scalar(245, 245, 245), 1);

        cv::imshow(window, canvas);
        int key = cv::waitKeyEx(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        } else if (key == 'a' || key == 'A' || key == 2424832) {
            manual = Signal::LeftTurn;
        } else if (key == 'd' || key == 'D' || key == 2555904) {
            manual = Signal::RightTurn;
        } else if (key == 's' || key == 'S' || key == 32) {
            manual = Signal::Stop;
        } else if (key == 'w' || key == 'W' || key == 2490368) {
            manual = Signal::Straight;
        } else if (key == 'v' || key == 'V') {
            manual.reset();
        }
    }
    return 0;
}
