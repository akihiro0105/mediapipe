licenses(["notice"])

package(default_visibility=[
    "//visibility:public",
])

cc_library(
    name="send_landmark_main",
    srcs=["send_landmark_main.cc"],
    deps=[
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
         "//mediapipe/framework/formats:landmark_cc_proto",
         ":udp_network_sender",
    ],
)

cc_binary(
    name="holistic_send_landmark",
    deps=[
        ":send_landmark_main",
        "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",
        "//mediapipe/calculators/core:pass_through_calculator",
    ],
)

cc_library(
    name="udp_network_sender",
    srcs=["udp_network_sender.cc"],
    hdrs=["udp_network_sender.h"],
)