cmake_minimum_required(VERSION 3.20)
project(hand_signal_car LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv/build" CACHE PATH "OpenCV config directory")
find_package(OpenCV REQUIRED)

add_executable(hand_signal_car
    src/main.cpp
    src/pose_estimator.cpp
    src/road_scene.cpp
)

target_include_directories(hand_signal_car PRIVATE ${OpenCV_INCLUDE_DIRS} src)
target_link_libraries(hand_signal_car PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(hand_signal_car PRIVATE /W3 /EHsc)
endif()

set(MODEL_FILE "${CMAKE_SOURCE_DIR}/models/yolov8n-pose.onnx")
set(GUIDE_FILE "${CMAKE_SOURCE_DIR}/top-hand-signals-driving-1166-image.jpg")

add_custom_command(TARGET hand_signal_car POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:hand_signal_car>/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_FILE}" "$<TARGET_FILE_DIR:hand_signal_car>/models/yolov8n-pose.onnx"
)

if(EXISTS "${GUIDE_FILE}")
    add_custom_command(TARGET hand_signal_car POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GUIDE_FILE}" "$<TARGET_FILE_DIR:hand_signal_car>/hand_signals.jpg"
    )
endif()

file(GLOB OPENCV_RUNTIME_DLLS
    "${OpenCV_DIR}/x64/vc16/bin/*.dll"
    "${OpenCV_DIR}/x64/vc17/bin/*.dll"
)
foreach(dll ${OPENCV_RUNTIME_DLLS})
    add_custom_command(TARGET hand_signal_car POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:hand_signal_car>"
    )
endforeach()
