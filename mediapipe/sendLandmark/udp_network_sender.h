#ifndef UDP_NETWORK_SENDER_H_
#define UDP_NETWORK_SENDER_H_

void UDP_Network_Init();
void UDP_Network_SendMessage(std::string message);
void UDP_Network_Send(int w, int h, int pose_size, float *pose_point, int left_size, float *left_point, int right_size, float *right_point, int face_size, float *face_point);
void UDP_Network_Close();

std::string setString_int(int data);
std::string setString_float(float data);
std::string setString_float_list(int size, float *point);

#endif // UDP_NETWORK_SENDER_H_