#include "rfcn_predict.h"
#include <algorithm>

template <typename T> void parse_vector(YAML::Node node, vector<T> &vec) {
  for (std::size_t i = 0; i < node.size(); i++) {
    vec.push_back(node[i].as<T>());
  }
}

string basename(const string path) {
  vector<string> strs;
  boost::split(strs, path, boost::is_any_of("/"));
  return strs[strs.size() - 1];
}

string remove_ent(const string s) {
  vector<string> strs;
  boost::split(strs, s, boost::is_any_of("."));
  return strs[0];
}

RFCN_detector::RFCN_detector(const string &model_file, const string &trained_file,
                           const bool use_GPU, const int batch_size,
                           const int devide_id, YAML::Node config) {
  if (use_GPU) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(devide_id);
    useGPU_ = true;
  } else {
    Caffe::set_mode(Caffe::CPU);
    useGPU_ = false;
  }

  /* Set batchsize */
  batch_size_ = batch_size;

  /* Load the network. */
  cout << "loading " << model_file << endl;
  net_.reset(new Net<float>(model_file, TEST));
  cout << "loading " << trained_file << endl;
  net_->CopyTrainedLayersFrom(trained_file);

  Blob<float> *input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  sliding_window_stride_ = config["FEAT_STRIDE"].as<int>();
  mean_values_.push_back(102.9801);
  mean_values_.push_back(115.9465);
  mean_values_.push_back(122.7717);
  scale_ = config["TRAIN"]["DATA_AUG"]["NORMALIZER"].as<float>();
}

void RFCN_detector::set_input_geometry(int height, int width) {
  input_geometry_.height = height;
  input_geometry_.width = width;
}
// predict single frame forward function
vector<boost::shared_ptr<Blob<float>>>
RFCN_detector::forward(vector<cv::Mat> imgs) {
  Blob<float> *input_layer = net_->input_blobs()[0];
  Blob<float> *input_layer_info = net_->input_blobs()[1];
  input_geometry_.height = imgs[0].rows;
  input_geometry_.width = imgs[0].cols;
  input_layer->Reshape(batch_size_, num_channels_, input_geometry_.height,
                       input_geometry_.width);
  input_layer_info->Reshape(1,2,1,1);

  int dim = input_geometry_.height * input_geometry_.width;
  float *input_data = input_layer->mutable_cpu_data();
  float* input_data_info = input_layer_info->mutable_cpu_data();
  for (int i = 0; i < imgs.size(); i++) {

    Mat sample = imgs[i];

    if ((sample.rows != input_geometry_.height) ||
        (sample.cols != input_geometry_.width)) {
      resize(sample, sample,
             Size(input_geometry_.width, input_geometry_.height));
    }

    sample.convertTo(sample, CV_32FC3);
    sample = (sample - Scalar(102.9801, 115.9465, 122.7717)) / (float)scale_;

    vector<cv::Mat> input_channels;
    cv::split(sample, input_channels);
    float *input_imgdata = NULL;
    for (int i = 0; i < num_channels_; i++) {
      input_imgdata = (float *)input_channels[i].data;
      memcpy(input_data, input_imgdata, sizeof(float) * dim);
      input_data += dim;
    }
  }
  input_data_info[0] = input_geometry_.height;
  input_data_info[1] = input_geometry_.width;

  net_->Reshape();

  net_->ForwardPrefilled();
  if (useGPU_) {
    cudaDeviceSynchronize();
  }

  vector<boost::shared_ptr<Blob<float>>> outputs;
  boost::shared_ptr<Blob<float>> rois =
      net_->blob_by_name(string("rois"));
  boost::shared_ptr<Blob<float>> cls_prob =
      net_->blob_by_name(string("cls_prob"));
  boost::shared_ptr<Blob<float>> bbox_pred =
      net_->blob_by_name(string("bbox_pred"));
  outputs.push_back(rois);
  outputs.push_back(cls_prob);
  outputs.push_back(bbox_pred);

  return outputs;
}

void RFCN_detector::nms(vector<struct Bbox> &p, float threshold) {

  sort(p.begin(), p.end(), mycmp);
  for (int i = 0; i < p.size(); i++) {
    if (p[i].deleted)
      continue;
    for (int j = i + 1; j < p.size(); j++) {

      if (!p[j].deleted) {
        cv::Rect intersect = p[i].rect & p[j].rect;
        float iou = intersect.area() * 1.0 /
                    (p[i].rect.area() + p[j].rect.area() - intersect.area());
        if (iou > threshold) {
          p[j].deleted = true;
        }
      }
    }
  }
}

void RFCN_detector::get_input_size(int &batch_size, int &num_channels,
                                  int &height, int &width) {
  batch_size = batch_size_;
  num_channels = num_channels_;
  height = input_geometry_.height;
  width = input_geometry_.width;
}

void RFCN_detector::get_detection(vector<boost::shared_ptr<Blob<float> > > &outputs, vector<struct Bbox> &dets, float score_thres, float nms_thres, float resize_height, float resize_width, float resize_ratio){
	const float* rois = outputs[0]->cpu_data();
	const float* cls_prob = outputs[1]->cpu_data();
	const float* pred_bbox = outputs[2]->cpu_data();
	vector<struct Bbox> bbs;
	int box_num= outputs[0]->num();
	for(int i=0;i<box_num;++i){
		float score= cls_prob[i*2+1];
		if (score < score_thres) continue;
		float x1=rois[5*i+1];	
		float y1=rois[5*i+2];	
		float x2=rois[5*i+3];	
		float y2=rois[5*i+4];	
		float width= x2-x1+1;
		float height= y2-y1+1;
		float cx= x1+ 0.5*width;
		float cy= y1+ 0.5*height;
		float dx= pred_bbox[i*8+4];
		float dy= pred_bbox[i*8+5];
		float dw= pred_bbox[i*8+6];
		float dh= pred_bbox[i*8+7];
		float lx = cx+dx*width - 0.5* exp(dw)*width;
		float ly = cy+dy*height - 0.5* exp(dh)*height;
		float rx = cx+dx*width + 0.5* exp(dw)*width;
		float ry = cy+dy*height + 0.5* exp(dh)*height;
		int xmin = std::max(0.0f, lx)*resize_ratio;
		int xmax = std::min(resize_width, rx)*resize_ratio;
		int ymin = std::max(0.0f, ly)*resize_ratio;
		int ymax = std::min(resize_height, ry)*resize_ratio;
		if ((ymax - ymin) * (xmax - xmin) < 5*5) continue;
		struct Bbox bb;
		bb.rect= Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1);
		bb.confidence = score;
		bb.deleted= false;
		bbs.push_back(bb);
	}

  if (bbs.size() != 0) nms(bbs, nms_thres);

  dets.clear();
  for(int i = 0; i < bbs.size(); i++) {                                                           
	  if(!bbs[i].deleted)  dets.push_back(bbs[i]);
  }   

}

void DetectionForVideo(string &model_file, string &trained_file, YAML::Node &config, string &img_file, bool x_show) {

  vector<int> sliding_window_width;
  vector<int> sliding_window_height;
  float score_thres;
  float nms_thres;

  parse_vector<int>(config["ANCHOR_GENERATOR"]["SLIDING_WINDOW_WIDTH"],
                    sliding_window_width);
  parse_vector<int>(config["ANCHOR_GENERATOR"]["SLIDING_WINDOW_HEIGHT"],
                    sliding_window_height);
  score_thres = config["TEST"]["THRESH"].as<float>();
  nms_thres = config["TEST"]["NMS"].as<float>();

  std::ifstream infile(img_file.c_str());
  std::string img_path, anno_path;

  cout << "loading model..." << endl;
  RFCN_detector rfcn_det(model_file, trained_file, true, 1, 0, config);

  struct timeval start, end;

  while (infile >> img_path >> anno_path) {
    cv::Mat frame;

    frame = imread(img_path);
    if (frame.empty()) {
      cout << "Wrong Image" << endl;
      continue;
    }

    int output_width = frame.cols;
    int output_height = frame.rows;
    cout << output_width << " " << output_height << endl;

    int batch_size = 0, num_channels = 0, resize_width = 0, resize_height = 0;
    rfcn_det.get_input_size(batch_size, num_channels, resize_height,
                           resize_width);
    resize_width = (int)output_width * resize_height / output_height;

    cout << "input size: (" << resize_height << ", " << resize_width << ")"
         << endl;
    float enlarge_ratio = output_height * 1.0 / resize_height;

    Mat img = frame.clone();
    Mat norm_img;
	cv::resize(img, norm_img, cv::Size(resize_width, resize_height));

    vector<Mat> images;
    images.push_back(norm_img);

    gettimeofday(&start, NULL);
    vector<boost::shared_ptr<Blob<float>>> outputs = rfcn_det.forward(images);
    gettimeofday(&end, NULL);
	cout<<"forward time: "<< double(end.tv_sec - start.tv_sec)*1000.0 + double(end.tv_usec - start.tv_usec) / 1.0e3 <<endl;                                                                         

    gettimeofday(&start, NULL);
    vector<struct Bbox> result;
	rfcn_det.get_detection(outputs, result, score_thres, nms_thres, resize_height, resize_width, enlarge_ratio);
    gettimeofday(&end, NULL);
	cout<<"get_detection time: "<< double(end.tv_sec - start.tv_sec)*1000.0 + double(end.tv_usec - start.tv_usec) / 1.0e3 <<endl;                                                                         


    if (x_show) {
      char str_info[100];

      for (int bbox_id = 0; bbox_id < result.size(); bbox_id++) {
        rectangle(frame, result[bbox_id].rect, Scalar(0, 255, 0),
                  1.0);
        sprintf(str_info, "%.3f", result[bbox_id].confidence);
        string prob_info(str_info);
        putText(frame, prob_info, Point(result[bbox_id].rect.x, result[bbox_id].rect.y),
                CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255));
      }

	 cv::namedWindow("Cam");
	 cv::imshow("Cam", frame);
	 int key=cv::waitKey();
	 if (key==-1 || key== 27) exit(0);
    }
  }
}

DEFINE_bool(show, false, "whether show");

int main(int argc, char **argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("rfcn prediction\n"
                          "Usage:\n"
                          "    prediction [-show] deploy "
                          "caffemodel img_file config_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], " ");
    return 1;
  }

  const bool is_show = FLAGS_show;

  string model_file = argv[1];
  string trained_file = argv[2];
  string src = argv[3];
  string config_file = argv[4];

  YAML::Node config = YAML::LoadFile(config_file);

  google::InitGoogleLogging(argv[0]);

  DetectionForVideo(model_file, trained_file, config, src, is_show);
}
