#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <chrono>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"

#pragma comment(lib,"ws2_32.lib")

struct imgInfo {
    int img_cols;
    int img_rows;
    int img_channels;
};


int initClient()
{

    WORD sockVersion = MAKEWORD(2, 2);
    WSADATA wsaData;//WSADATA结构体变量的地址值

    if (WSAStartup(sockVersion, &wsaData) != 0)
    {
        std::cout << "WSAStartup() error!" << std::endl;
        return 0;
    }

    int client_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (client_socket == -1) {
        printf("client error!\n");
        return -1;
    }

    struct sockaddr_in client_addr;
    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sin_family = AF_INET;

    client_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    client_addr.sin_port = htons(6789);

    printf("waiting for starting Server...\n");

    while (connect(client_socket, (sockaddr*)&client_addr, sizeof(client_addr)) == -1);

    printf("connecting Server successfully\n");
    return client_socket;
    
}


int sendImgInfo(int new_client_socket, imgInfo* sendInfo) {

    int ret = send(new_client_socket, (char*)sendInfo, sizeof(imgInfo), 0);

    return ret;
}


int sendImg(int new_client_socket, cv::Mat & img) {


    int total = img.rows * img.cols * img.channels();
    char* data = new char[total];


    memcpy(data, img.data, sizeof(unsigned char) * total);

    int send_size = 0;
    send_size = send(new_client_socket, data, sizeof(unsigned char) * total, 0);
    if (send_size != total) {
        printf("client send img failed!\n");
        delete[] data;
        return 0;
    }
    delete[] data;
    return 1;
}


int recvResult(int new_client_socket) {

    int total = sizeof(int);

    char* data = new char[total];

    int sum = recv(new_client_socket, data, total, 0);

    if (sum < 0) {
        printf("client recieve data failed!\n");
        return -1;
    }

    int ans = *data;
    delete[] data;

    return ans;

}


#include <TlHelp32.h>
 
DWORD GetProcessidFromName(LPCTSTR processName)
{
    PROCESSENTRY32 pe;
    DWORD id = 0;
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    pe.dwSize = sizeof(PROCESSENTRY32);
    if (!Process32First(hSnapshot, &pe))
        return 0;
    while (true)
    {
        pe.dwSize = sizeof(PROCESSENTRY32);
        if (Process32Next(hSnapshot, &pe) == FALSE)
            break;


        if (wcscmp(pe.szExeFile, processName) == 0)
        {
            id = pe.th32ProcessID;
            break;
        }
    }
    CloseHandle(hSnapshot);
    return id;

}



int main() {

    if (GetProcessidFromName(L"vstestonnx.exe") == 0) {

        
        system("start vstestonnx.exe");
    
    };

    auto start = std::chrono::steady_clock::now();

    int client_socket = initClient();

    cv::Mat org_img = cv::imread("../test.jpg");

    imgInfo info;
    info.img_rows = org_img.rows;
    info.img_cols = org_img.cols;
    info.img_channels = org_img.channels();

    sendImgInfo(client_socket,&info);
    sendImg(client_socket, org_img);
    int ans = recvResult(client_socket);

    std::cout<<"ans = " << ans <<std::endl;

    auto end = std::chrono::steady_clock::now();
    std::cout <<(end - start).count() * 1e-6 << "ms" << std::endl;


    system("pause");


}
