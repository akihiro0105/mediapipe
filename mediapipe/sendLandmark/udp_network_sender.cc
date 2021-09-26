#pragma comment(lib, "ws2_32.lib")

#include <WinSock2.h>
#include <iostream>
#include <string>
#include <sstream>

#pragma warning(disable : 4996)

WSAData wsaData;
int sock;
struct sockaddr_in addr;

void UDP_Network_Init()
{
    WSACleanup();
    WSAStartup(MAKEWORD(2, 0), &wsaData);
    sock = socket(AF_INET, SOCK_DGRAM, 0);

    addr.sin_family = AF_INET;
    addr.sin_port = htons(50008);
    addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
}

void UDP_Network_SendMessage(std::string message)
{
    std::cout << message << std::endl;
    auto data = message.c_str();
    sendto(sock, data, strlen(data), 0, (struct sockaddr *)&addr, sizeof(addr));
}

std::string setString_int(int data)
{
    std::ostringstream ss;
    ss << data;
    return ss.str();
}

std::string setString_float(float data)
{
    std::ostringstream buf;
    buf << data;
    return buf.str();
}

std::string setString_float_list(int size, float *point)
{
    auto message = setString_int(size) + ",";
    for (size_t i = 0; i < size; i++)
    {
        message += setString_float(point[i * 3 + 0]) + ",";
        message += setString_float(point[i * 3 + 1]) + ",";
        message += setString_float(point[i * 3 + 2]) + ",";
    }
    return message;
}

void UDP_Network_Send(int w, int h, int pose_size, float *pose_point, int left_size, float *left_point, int right_size, float *right_point, int face_size, float *face_point)
{
    std::string message = "";
    message += setString_int(w) + ",";
    message += setString_int(h) + ",";
    message += setString_float_list(pose_size, pose_point);
    message += setString_float_list(left_size, left_point);
    message += setString_float_list(right_size, right_point);
    message += setString_float_list(face_size, face_point);
    message += "0";
    // send message
    auto data = message.c_str();
    sendto(sock, data, strlen(data), 0, (struct sockaddr *)&addr, sizeof(addr));
}

void UDP_Network_Close()
{
    closesocket(sock);
    WSACleanup();
}