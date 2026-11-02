#pragma once

// Driving signals made with the left arm, matching a standard hand-signal guide:
//   arm straight out to the side -> left turn
//   elbow bent, forearm pointing up -> right turn
//   elbow bent, forearm pointing down -> stop
enum class Signal { Straight, LeftTurn, RightTurn, Stop, Unknown };

inline const char* signalLabel(Signal signal) {
    switch (signal) {
    case Signal::LeftTurn: return "LEFT TURN";
    case Signal::RightTurn: return "RIGHT TURN";
    case Signal::Stop: return "STOP";
    case Signal::Straight: return "STRAIGHT";
    case Signal::Unknown: return "NO ARM";
    }
    return "STRAIGHT";
}
