#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <opencv2/opencv.hpp>

struct  Object{
  cv::Rect_<float> rect;
  int label;
  float prob;

};

class YOLO{

    public:

    std::vector<std::string> class_names = {"GW4_kai","GW4_he","GW7_kai","GW7_he",
                                        "GW10_kai","GW10_he","GW11_kai","GW11_he"};

    cv::Mat preprocessImg(const cv::Mat& img,const int & input_w, const int & input_h);
    void qsort_descent_inplace(std::vector<Object>&faceobjects,int left, int right);
    void qsort_descent_inplace(std::vector<Object>&faceobjects);
    float intersection_area(Object & a,Object&b) ;
    void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
    std::vector<Object> decode(float * results,const int & org_img_rows_,const int & org_img_cols_ );
    void drawObjects( cv::Mat & image ,std::vector<Object> & objects );
    void onnx_run(cv::Mat & img);

    const int BATCH_SIZE = 1;
    const int CHANNELS = 3;
    const int WIDTH = 640;
    const int HEIGHT = 640;
    const int BOXES = 25200;
    const int NUM = 13;

    const char* input_names[1] = {"images"};
    const char* output_names[1] = {"output0"};

    Ort::Env env;
    Ort::Session session_{env,"../best.onnx",Ort::SessionOptions{nullptr}};
    Ort::Value input_tensor_{nullptr};
    std::vector<int64_t> input_shape_{BATCH_SIZE, CHANNELS, HEIGHT, WIDTH};
    Ort::Value output_tensor_{nullptr};
    std::vector<int64_t> output_shape_{BATCH_SIZE, BOXES, NUM};
    Ort::RunOptions run_options;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

};



cv::Mat YOLO::preprocessImg(const cv::Mat& img,const int & input_w, const int & input_h){
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
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


void YOLO::qsort_descent_inplace(std::vector<Object>&faceobjects,int left, int right){
    int i = left;
    int j = right;
    float p = faceobjects[(left+right)/2].prob;
    while (i<=j){
        while (faceobjects[i].prob>p ){
            i++;
        }
        while (faceobjects[j].prob<p){
            j--;
        }
        if(i<=j){
            std::swap(faceobjects[i],faceobjects[j]);
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

void YOLO::qsort_descent_inplace(std::vector<Object>&faceobjects){
    if(faceobjects.empty()){
        return ;
    }
    qsort_descent_inplace(faceobjects,0,faceobjects.size()-1);
}

float YOLO::intersection_area(Object & a,Object&b) {
    cv::Rect_<float> inter = a.rect&b.rect;
    return inter.area();

}

void YOLO::nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
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

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


std::vector<Object> YOLO::decode(float * results,const int & org_img_rows_,const int & org_img_cols_ ){

    std::vector<Object> objects;
    for(int i = 0 ; i<BOXES;i++){
        if(results[NUM*i+4]>0.3){
            int l,r,t,b;
            float r_w = WIDTH/(org_img_cols_*1.0);
            float r_h = HEIGHT/(org_img_rows_*1.0);

            float x = results[NUM*i+0];
            float y = results[NUM*i+1];
            float w = results[NUM*i+2];
            float h = results[NUM*i+3];
            float score = results[NUM*i+4];

            if(r_h>r_w){
                l = x-w/2.0;
                r = x+w/2.0;
                t = y-h/2.0-(HEIGHT-r_w*org_img_rows_)/2;
                b = y+h/2.0-(HEIGHT-r_w*org_img_rows_)/2;
                l=l/r_w;
                r=r/r_w;
                t=t/r_w;
                b=b/r_w;
            }else{
                l = x-w/2.0-(WIDTH-r_h*org_img_cols_)/2;
                r = x+w/2.0-(WIDTH-r_h*org_img_cols_)/2;
                t = y-h/2.0;
                b = y+h/2.0;
                l=l/r_h;
                r=r/r_h;
                t=t/r_h;
                b=b/r_h;
            }
            int label_index = std::max_element(results+NUM*i+5,results+NUM*(i+1)) - (results+NUM*i+5);
         
            Object obj;
            obj.rect.x = std::max(l,0);
            obj.rect.y = std::max(t,0);
            obj.rect.width=r-l;
            obj.rect.height=b-t;
            if(obj.rect.x+obj.rect.width>org_img_cols_){
                obj.rect.width = org_img_cols_ - obj.rect.x;
            }
            if(obj.rect.y+obj.rect.height>org_img_rows_){
                obj.rect.height = org_img_rows_ - obj.rect.y;
            }
            obj.label = label_index;
            obj.prob = score;
            objects.push_back(obj);
            
        }

    }
    qsort_descent_inplace(objects);
    std::vector<int> picked;
    nms_sorted_bboxes(objects,picked,0.45);
    int count = picked.size();
    std::cout<<"count="<<count<<std::endl;
    std::vector<Object>obj_out(count);
    for(int i = 0 ; i <count ; ++i){
        obj_out[i] = objects[picked[i]];
    }
    return obj_out;
}



void YOLO::drawObjects( cv::Mat & image ,std::vector<Object> & objects ){

    for(int idx = 0 ; idx < objects.size(); idx ++){

            Object obj = objects[idx];
            
            if (obj.prob < 0.15)
                continue;

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::rectangle(image, obj.rect, cv::Scalar(151,255,0));

            char text[256];

            sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y ;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    }



}
void YOLO::onnx_run(cv::Mat & org_img){
    
    
    std::vector<float> input_image_(BATCH_SIZE*CHANNELS*WIDTH*HEIGHT,0);
    std::vector<float> results_(BATCH_SIZE*BOXES*NUM,0);

    cv::Mat pr_img = preprocessImg(org_img,WIDTH,HEIGHT);

    for(int i = 0 ; i < WIDTH * HEIGHT;i++){
            input_image_[i] = pr_img.at<cv::Vec3b>(i)[2]/255.0;
            input_image_[i+WIDTH * HEIGHT] = pr_img.at<cv::Vec3b>(i)[1]/255.0;
            input_image_[i+2*WIDTH * HEIGHT]=pr_img.at<cv::Vec3b>(i)[0]/255.0;
        }

    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                    input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                     output_shape_.data(), output_shape_.size());


    session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);


    std::vector<Object> objects = decode(results_.data(),org_img.rows,org_img.cols);

    drawObjects(org_img,objects);
    cv::imwrite("../test_out.jpg",org_img);

}



int main(){

    YOLO yolo;

    cv::Mat org_img = cv::imread("../test.jpg");
    yolo.onnx_run(org_img);
  

}
