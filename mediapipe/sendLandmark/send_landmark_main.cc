// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/sendLandmark/udp_network_sender.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

constexpr char kOutputPoseLandmarks[] = "output_pose_landmarks";
constexpr char kOutputLeftLandmarks[] = "output_left_hand_landmarks";
constexpr char kOutputRightLandmarks[] = "output_right_hand_landmarks";
constexpr char kOutputFaceLandmarks[] = "output_face_landmarks";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph(int cameraIndex, bool isView)
{
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: "
              << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Initialize the camera or load the video.";
    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video)
    {
        capture.open(absl::GetFlag(FLAGS_input_video_path));
    }
    else
    {
        capture.open(cameraIndex);
    }
    RET_CHECK(capture.isOpened());

    cv::VideoWriter writer;
    const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
    if (!save_video)
    {
        if (isView)
        {
            cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
            capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            capture.set(cv::CAP_PROP_FPS, 30);
#endif
        }
    }

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_pose, graph.AddOutputStreamPoller(kOutputPoseLandmarks));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_left, graph.AddOutputStreamPoller(kOutputLeftLandmarks));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_right, graph.AddOutputStreamPoller(kOutputRightLandmarks));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_face, graph.AddOutputStreamPoller(kOutputFaceLandmarks));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "Start grabbing and processing frames.";
    bool grab_frames = true;

    UDP_Network_Init();
    LOG(INFO) << "Start UDP network sender.";

    while (grab_frames)
    {
        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty())
        {
            if (!load_video)
            {
                LOG(INFO) << "Ignore empty frames from camera.";
                continue;
            }
            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        if (!load_video)
        {
            cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
        }

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release())
                              .At(mediapipe::Timestamp(frame_timestamp_us))));

        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet packet;

        mediapipe::Packet packet_pose;
        int pose_size = 0;
        float pose_point[33 * 3];
        mediapipe::Packet packet_left;
        int left_size = 0;
        float left_hand_point[21 * 3];
        mediapipe::Packet packet_right;
        int right_size = 0;
        float right_hand_point[21 * 3];
        mediapipe::Packet packet_face;
        int face_size = 0;
        float face_point[468 * 3];

        if (!poller.Next(&packet))
            break;
        auto &output_frame = packet.Get<mediapipe::ImageFrame>();

        std::ostringstream w, h;
        w << output_frame.Width();
        h << output_frame.Height();
        std::string message = "w:" + w.str() + "h:" + h.str();

        if (poller_pose.QueueSize() > 0)
        {
            if (!poller_pose.Next(&packet_pose))
                break;
            auto &output_pose = packet_pose.Get<mediapipe::NormalizedLandmarkList>();
            auto list = output_pose.landmark();
            pose_size = list.size();
            for (size_t i = 0; i < list.size(); i++)
            {
                pose_point[i * 3 + 0] = list[i].x();
                pose_point[i * 3 + 1] = list[i].y();
                pose_point[i * 3 + 2] = list[i].z();
            }
            std::ostringstream ss;
            ss << list.size();
            message += "pose:" + ss.str();
        }

        if (poller_left.QueueSize() > 0)
        {
            if (!poller_left.Next(&packet_left))
                break;
            auto &output_left = packet_left.Get<mediapipe::NormalizedLandmarkList>();
            auto list = output_left.landmark();
            left_size = list.size();
            for (size_t i = 0; i < list.size(); i++)
            {
                left_hand_point[i * 3 + 0] = list[i].x();
                left_hand_point[i * 3 + 1] = list[i].y();
                left_hand_point[i * 3 + 2] = list[i].z();
            }
            std::ostringstream ss;
            ss << list.size();
            message += "left:" + ss.str();
        }

        if (poller_right.QueueSize() > 0)
        {
            if (!poller_right.Next(&packet_right))
                break;
            auto &output_right = packet_right.Get<mediapipe::NormalizedLandmarkList>();
            auto list = output_right.landmark();
            right_size = list.size();
            for (size_t i = 0; i < list.size(); i++)
            {
                right_hand_point[i * 3 + 0] = list[i].x();
                right_hand_point[i * 3 + 1] = list[i].y();
                right_hand_point[i * 3 + 2] = list[i].z();
            }
            std::ostringstream ss;
            ss << list.size();
            message += "right:" + ss.str();
        }

        if (poller_face.QueueSize() > 0)
        {
            if (!poller_face.Next(&packet_face))
                break;
            auto &output_face = packet_face.Get<mediapipe::NormalizedLandmarkList>();
            auto list = output_face.landmark();
            face_size = list.size();
            for (size_t i = 0; i < list.size(); i++)
            {
                face_point[i * 3 + 0] = list[i].x();
                face_point[i * 3 + 1] = list[i].y();
                face_point[i * 3 + 2] = list[i].z();
            }
            std::ostringstream ss;
            ss << list.size();
            message += "face:" + ss.str();
        }
        UDP_Network_Send(output_frame.Width(), output_frame.Height(), pose_size, pose_point, left_size, left_hand_point, right_size, right_hand_point, face_size, face_point);
        //UDP_Network_SendMessage(message);

        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        if (save_video)
        {
            if (!writer.isOpened())
            {
                LOG(INFO) << "Prepare video writer.";
                writer.open(absl::GetFlag(FLAGS_output_video_path),
                            mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                            capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
                RET_CHECK(writer.isOpened());
            }
            writer.write(output_frame_mat);
        }
        else
        {
            if (isView)
            {
                cv::imshow(kWindowName, output_frame_mat);
            }
            // Press any key to exit.
            const int pressed_key = cv::waitKey(5);
            if (pressed_key >= 0 && pressed_key != 255)
                grab_frames = false;
        }
    }
    UDP_Network_Close();
    LOG(INFO) << "Close UDP network sender.";

    LOG(INFO) << "Shutting down.";
    if (writer.isOpened())
        writer.release();
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    auto newArgc = argc - 2;
    int deviceNum = atoi(argv[newArgc]);
    bool isView = false;
    if (atoi(argv[newArgc + 1]) == 1)
    {
        isView = true;
    }

    absl::ParseCommandLine(newArgc, argv);
    absl::Status run_status = RunMPPGraph(deviceNum, isView);
    if (!run_status.ok())
    {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    else
    {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
