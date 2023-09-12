#include "onnxruntime_cxx_api.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"

//std::vector<std::string> class_names = { "GW4_kai","GW4_he","GW7_kai","GW7_he",
//                                     "GW10_kai","GW10_he","GW11_kai","GW11_he" };

const int BATCH_SIZE = 1;
const int CHANNELS   = 3;
const int WIDTH      = 640;
const int HEIGHT     = 640;
const int BOXES      = 25200;
const int NUM        = 13;

struct  Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

cv::Mat preprocessImg(const cv::Mat& img, const int& input_w, const int& input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}
void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    while (i <= j) {
        while (faceobjects[i].prob > p) {
            i++;
        }
        while (faceobjects[j].prob < p) {
            j--;
        }
        if (i <= j) {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}
void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty()) {
        return;
    }
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}
float intersection_area(Object& a, Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }
    for (int i = 0; i < n; i++)
    {
        Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}
std::vector<Object> decode(float* results, const int& org_img_rows_, const int& org_img_cols_) {
    std::vector<Object> objects;
    for (int i = 0; i < BOXES; i++) {
        if (results[NUM * i + 4] > 0.5) {
            int l, r, t, b;
            float r_w = WIDTH / (org_img_cols_ * 1.0);
            float r_h = HEIGHT / (org_img_rows_ * 1.0);
            float x = results[NUM * i + 0];
            float y = results[NUM * i + 1];
            float w = results[NUM * i + 2];
            float h = results[NUM * i + 3];
            float score = results[NUM * i + 4];
            if (r_h > r_w) {
                l = x - w / 2.0;
                r = x + w / 2.0;
                t = y - h / 2.0 - (HEIGHT - r_w * org_img_rows_) / 2;
                b = y + h / 2.0 - (HEIGHT - r_w * org_img_rows_) / 2;
                l = l / r_w;
                r = r / r_w;
                t = t / r_w;
                b = b / r_w;
            }
            else {
                l = x - w / 2.0 - (WIDTH - r_h * org_img_cols_) / 2;
                r = x + w / 2.0 - (WIDTH - r_h * org_img_cols_) / 2;
                t = y - h / 2.0;
                b = y + h / 2.0;
                l = l / r_h;
                r = r / r_h;
                t = t / r_h;
                b = b / r_h;
            }
            int label_index = std::max_element(results + NUM * i + 5, results + NUM * (i + 1)) - (results + NUM * i + 5);
            Object obj;
            obj.rect.x = std::max(l, 0);
            obj.rect.y = std::max(t, 0);
            obj.rect.width = r - l;
            obj.rect.height = b - t;
            if (obj.rect.x + obj.rect.width > org_img_cols_) {
                obj.rect.width = org_img_cols_ - obj.rect.x;
            }
            if (obj.rect.y + obj.rect.height > org_img_rows_) {
                obj.rect.height = org_img_rows_ - obj.rect.y;
            }
            obj.label = label_index;
            obj.prob = score;
            objects.push_back(obj);
        }
    }
    qsort_descent_inplace(objects);
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.45);
    int count = picked.size();
    //std::cout << "count=" << count << std::endl;
    std::vector<Object>obj_out(count);
    for (int i = 0; i < count; ++i) {
        obj_out[i] = objects[picked[i]];
    }
    return obj_out;
}


#include <winsock2.h>
#pragma comment(lib,"ws2_32.lib")

struct imgInfo {
    int img_cols;
    int img_rows;
    int img_channels;
};

int initServer()
{
    WORD sockVersion = MAKEWORD(2, 2);
    WSADATA wsaData;//WSADATA结构体变量的地址值

    if (WSAStartup(sockVersion, &wsaData) != 0)
    {
        std::cout << "WSAStartup() error!" << std::endl;
        return 0;
    }

    int server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(6789);

    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
       
        std::cout << "bind failed!" << std::endl;
        return -1;
    }

    if (listen(server_socket, 100) == -1){

        std::cout << "listen failed!" << std::endl;

        return -1;
    }
    return server_socket;
}


int connect2Client(int server_socket)
{
 
    struct sockaddr_in remote_addr;
    int remote_addr_size = sizeof(remote_addr);

    std::cout << "waiting for client connect..." << std::endl;
    int new_server_socket = accept(server_socket, (struct sockaddr*)&remote_addr, &remote_addr_size);
    
    if (new_server_socket < 0)
    {
        std::cout<<"connect failed"<<std::endl;
        return -1;
    }


    std::cout << "connecting client successfully !" << std::endl;

    return new_server_socket;
}


int recvInfo(int new_server_socket, imgInfo * info)
{
    int total = sizeof(imgInfo);

    int sum = recv(new_server_socket, (char*)info, total, 0);
    if (sum < 0)
    {
        std::cout << "Server Recieve imgInfo Failed!" << std::endl;
        return 0;
    }
    while (sum != total)
    {
        char* temp_data = new char[total - sum];
        int length = recv(new_server_socket, temp_data, total - sum, 0);
        if (length > 0)
        {
            memcpy(info + sum, temp_data, length);
            sum += length;
        }
        delete[] temp_data;
        if (length <= 0)
        {
            std::cout << "Server Recieve imgInfo Failed!" << std::endl;
            return 0;
        }
    }

    return 1;
}


int recvImg(int new_server_socket, imgInfo * info, cv::Mat&img)
{
    int img_height = info->img_rows;
    int img_width = info->img_cols;
    int channels = info->img_channels;
    int total = img_height * img_width * channels;
    char* data = new char[total];

    int sum = recv(new_server_socket, data, total, 0);


    if (sum <= 0)
    {
        std::cout << "Server Recieve img Failed!" << std::endl;
        return 0;
    }
    while (sum != total)
    {
        char* temp_data = new char[total - sum];
        int length = recv(new_server_socket, temp_data, total - sum, 0);
        if (length > 0)
        {
            memcpy(data + sum, temp_data, length);
            sum += length;
        }
        delete[] temp_data;
        if (length <= 0)
        {
            std::cout << "Server Recieve img Failed!" << std::endl;
            return 0;
        }
    }



    img.create(cv::Size(img_width, img_height), CV_8UC3);


    memcpy(img.data, data, img_height * img_width * channels);

    memset(data, 0, sizeof(char) * img_height * img_width * channels);
    delete[] data;
    return 1;
}

int sendResult(int new_server_socket, int* res)
{
    int send_length = 0;
    int total = sizeof(int);

    send_length = send(new_server_socket,(char *) res, total, 0);
    if (send_length != total)
    {
        std::cout << "Send RES Inform Failed!" << std::endl;
        return 0;
    }
    return 1;
}



int main() {

    int fd = initServer();

    Ort::Env env;
//  local test
//  Ort::Session session_{ env, L"../best.onnx", Ort::SessionOptions{nullptr} };

    Ort::Session session_{ env, L"../vstest2019/vstestonnx/best.onnx", Ort::SessionOptions{nullptr} };
    Ort::Value input_tensor_{ nullptr };
    std::vector<int64_t> input_shape_{ BATCH_SIZE, CHANNELS, HEIGHT, WIDTH };
    Ort::Value output_tensor_{ nullptr };
    std::vector<int64_t> output_shape_{ BATCH_SIZE, BOXES, NUM };
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    const char* input_names[] = { "images" };
    const char* output_names[] = { "output0" };
    Ort::RunOptions run_options;

    std::vector<float> input_image_(BATCH_SIZE * CHANNELS * WIDTH * HEIGHT, 0);
    std::vector<float> results_(BATCH_SIZE * BOXES * NUM, 0);


    while (true) {
    
        int server_socket = connect2Client(fd);

        imgInfo info;
        cv::Mat org_img;

        recvInfo(server_socket, &info);
        recvImg(server_socket, &info, org_img);


        cv::Mat pr_img = preprocessImg(org_img, WIDTH, HEIGHT);
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            input_image_[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            input_image_[i + WIDTH * HEIGHT] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            input_image_[i + 2 * WIDTH * HEIGHT] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }

        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
            input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
            output_shape_.data(), output_shape_.size());


        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        std::vector<Object> objects = decode(results_.data(), org_img.rows, org_img.cols);
        std::cout << " objects_size =  " << objects.size() << std::endl;

        int ans = 1;

        if (!objects.empty()) {

            int label = objects[0].label;
            if (label == 0 || label == 2 || label == 4 || label == 6)  ans = 0;
            if (label == 1 || label == 3 || label == 5 || label == 7)  ans = 1;


        }

        sendResult(server_socket, &ans);
    
    }

}
