#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include <iostream>
#include <caffe/caffe.hpp>
#include <iosfwd>
#include <memory>
#include <ctime>
#include <cassert>
#include <algorithm>
#include <ctype.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include <yaml-cpp/yaml.h>
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace caffe;


template<typename T>
void parse_vector(YAML::Node node, vector<T> & vec);

struct Bbox {
	int size_index;
	float confidence;
	Rect rect;
	bool deleted;

};
bool mycmp(struct Bbox b1, struct Bbox b2) {
	return b1.confidence > b2.confidence;
}

enum TASKtype { ALL, FACE, PD};

class RPN_detector {
public:
	RPN_detector(const string& model_file,
			const string& trained_file,
			const bool use_GPU,
			const int batch_size,
			const int device_id, YAML::Node node);

	vector<boost::shared_ptr<Blob<float> > > forward(vector<Mat> imgs, TASKtype TASK);
	void nms(vector<struct Bbox>& p, float threshold);
	vector<struct Bbox> get_detection(vector<Mat> images, vector<boost::shared_ptr<Blob<float> > >& output,int sliding_window_stride, vector<int> anchor_width, vector<int> anchor_height,  
			vector<float> rpn_thres, float nms_thres, float enlarge_ratiow, float enlarge_ratioh);
	void get_input_size(int& batch_size, int& num_channels, int& height, int& width);
	void set_input_geometry(int width, int height);

private:
	boost::shared_ptr<Net<float> > net_;
	int batch_size_;
	int num_channels_;
	cv::Size input_geometry_;	
	bool useGPU_;
	
	vector<float> mean_values_;
	float scale_;
};
