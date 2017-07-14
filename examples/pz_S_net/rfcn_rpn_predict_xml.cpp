#include "rfcn_rpn_predict.h"

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

RPN_detector::RPN_detector(const string &model_file, const string &trained_file,
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
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

  Blob<float> *input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  sliding_window_stride_ = config["FEAT_STRIDE"].as<int>();
  mean_values_.push_back(102.9801);
  mean_values_.push_back(115.9465);
  mean_values_.push_back(122.7717);
  scale_ = config["TRAIN"]["DATA_AUG"]["NORMALIZER"].as<float>();
}

void RPN_detector::set_input_geometry(int height, int width) {
  input_geometry_.height = height;
  input_geometry_.width = width;
}
// predict single frame forward function
vector<boost::shared_ptr<Blob<float>>>
RPN_detector::forward(vector<cv::Mat> imgs) {
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_geometry_.height = imgs[0].rows;
  input_geometry_.width = imgs[0].cols;
  input_layer->Reshape(batch_size_, num_channels_, input_geometry_.height,
                       input_geometry_.width);

  int dim = input_geometry_.height * input_geometry_.width;
  float *input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < imgs.size(); i++) {

    Mat sample;
    Mat img = imgs[i];

    if (img.channels() == 3 && num_channels_ == 1)
      cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cvtColor(img, sample, CV_RGBA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;

    if ((sample.rows != input_geometry_.height) ||
        (sample.cols != input_geometry_.width)) {
      resize(sample, sample,
             Size(input_geometry_.width, input_geometry_.height));
    }

    sample.convertTo(sample, CV_32FC3);
    // sample=(sample-cv::mean(sample))/(float)scale_;
    // sample=(sample-mean_values_)/(float)scale_;
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

  net_->ForwardPrefilled();
  if (useGPU_) {
    cudaDeviceSynchronize();
  }

  vector<boost::shared_ptr<Blob<float>>> outputs;
  boost::shared_ptr<Blob<float>> rpn_cls_score =
      net_->blob_by_name(string("rpn_cls_score"));
  boost::shared_ptr<Blob<float>> rpn_bbox_pred =
      net_->blob_by_name(string("rpn_bbox_pred"));
  outputs.push_back(rpn_cls_score);
  outputs.push_back(rpn_bbox_pred);

  return outputs;
}

void RPN_detector::nms(vector<struct Bbox> &p, float threshold) {

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

void RPN_detector::get_input_size(int &batch_size, int &num_channels,
                                  int &height, int &width) {
  batch_size = batch_size_;
  num_channels = num_channels_;
  height = input_geometry_.height;
  width = input_geometry_.width;
}

vector<struct Bbox> RPN_detector::get_detection(
    vector<Mat> images, vector<boost::shared_ptr<Blob<float>>> &outputs,
    vector<int> sliding_window_width, vector<int> sliding_window_height,
    vector<float> rpn_thres, float nms_thres, float enlarge_ratiow,
    float enlarge_ratioh) {

  assert(sliding_window_width.size() == sliding_window_height.size());

  int anchor_number = sliding_window_width.size();

  boost::shared_ptr<Blob<float>> cls = outputs[0];
  boost::shared_ptr<Blob<float>> reg = outputs[1];

  cls->Reshape(cls->num(), cls->channels(), cls->height(), cls->width());
  reg->Reshape(reg->num(), reg->channels(), reg->height(), reg->width());

  assert(cls->num() == reg->num());

  assert(cls->channels() == anchor_number * 2);
  assert(reg->channels() == anchor_number * 4);

  assert(cls->height() == reg->height());
  assert(cls->width() == reg->width());

  assert(images.size() == 1);

  vector<struct Bbox> vbbox;
  const float *cls_cpu = cls->cpu_data();
  const float *reg_cpu = reg->cpu_data();
  int img_height = images[0].rows;
  int img_width = images[0].cols;
  float w, h;
  int skip = cls->height() * cls->width();
  float log_thres[anchor_number];
  for (int i = 0; i < anchor_number; i++)
    log_thres[i] = log(rpn_thres[i] / (1.0 - rpn_thres[i]));
  float rect[4];
  for (int i = 0; i < cls->num(); i++) {
    for (int j = 0; j < anchor_number; j++) {
      h = sliding_window_height[j];
      w = sliding_window_width[j];
      for (int y_index = 0; y_index < int(img_height / sliding_window_stride_);
           y_index++) {
        int y = y_index * sliding_window_stride_ + sliding_window_stride_ / 2 -
                1 - h / 2;
        for (int x_index = 0; x_index < int(img_width / sliding_window_stride_);
             x_index++) {
          int x = x_index * sliding_window_stride_ +
                  sliding_window_stride_ / 2 - 1 - w / 2;

          float x0 = cls_cpu[j * skip + y_index * cls->width() + x_index];
          float x1 = cls_cpu[(j + anchor_number) * skip +
                             y_index * cls->width() + x_index];
          if (x1 - x0 > log_thres[j]) {
            rect[2] = exp(reg_cpu[(j * 4 + 2) * skip + y_index * reg->width() +
                                  x_index]) *
                      w;
            rect[3] = exp(reg_cpu[(j * 4 + 3) * skip + y_index * reg->width() +
                                  x_index]) *
                      h;

            rect[0] = reg_cpu[j * 4 * skip + y_index * reg->width() + x_index];
            rect[1] =
                reg_cpu[(j * 4 + 1) * skip + y_index * reg->width() + x_index];

            rect[0] = rect[0] * w + w / 2 - rect[2] / 2 + x;
            rect[1] = rect[1] * h + h / 2 - rect[3] / 2 + y;

            struct Bbox bbox;
            bbox.confidence = 1.0 / (1.0 + exp(x0 - x1));
            ;
            bbox.size_index = j;
            bbox.rect = Rect(rect[0], rect[1], rect[2], rect[3]);
            bbox.rect &= Rect(0, 0, img_width, img_height);
            bbox.deleted = false;
            vbbox.push_back(bbox);
          }
        }
      }
    }
  }

  if (vbbox.size() != 0)
    nms(vbbox, nms_thres);

  vector<struct Bbox> final_vbbox;

  for (int i = 0; i < vbbox.size(); i++) {
    if (!vbbox[i].deleted) {
      struct Bbox box = vbbox[i];
      float x = box.rect.x * enlarge_ratiow;
      float y = box.rect.y * enlarge_ratioh;
      float w = box.rect.width * enlarge_ratiow;
      float h = box.rect.height * enlarge_ratioh;
      box.rect.x = x;
      box.rect.y = y;
      box.rect.width = w;
      box.rect.height = h;
      final_vbbox.push_back(box);
    }
  }
  return final_vbbox;
}

// reading  images from one file
void createDocList(vector<string> &doc_list, const string path) {
  DIR *dpdf;
  struct dirent *epdf;
  dpdf = opendir(path.c_str());

  if (dpdf != NULL) {
    epdf = readdir(dpdf);
    while (epdf) {
      string name = string(epdf->d_name);
      epdf = readdir(dpdf);
      // for jpg format frame
      if (name[name.length() - 1] != 'g')
        continue;
      doc_list.push_back(path + string(name));
    }
    closedir(dpdf);
  } else {
    cout << "the path is empty" << endl;
  }
}

void createDocList_R(vector<string> &doc_list, const string path0) {
  cout << "NEW DIR: " << path0 << endl;
  string path(path0);
  if (path[path.length() - 1] != '/')
    path.append("/");
  DIR *pDir;
  struct dirent *ent;
  pDir = opendir(path.c_str());

  if (pDir != NULL) {
    while ((ent = readdir(pDir)) != NULL) {
      string name = string(ent->d_name);
      if (ent->d_type == 4) {
        if (name[0] == '.')
          continue;
        createDocList_R(doc_list, path + name);
      } else {
        // for jpg format frame
        if (name[name.length() - 1] != 'g')
          continue;
        doc_list.push_back(path + name);
      }
    }
    closedir(pDir);
  } else {
    cout << "the path is empty" << endl;
  }
}

void createImgsList(vector<string> &imgList, const string img_file) {
  std::ifstream infile(img_file.c_str());
  std::string img_path, anno_path;
  while (infile >> img_path >> anno_path) {
    imgList.push_back(img_path);
  }
}

void DetectionForVideo(string &model_file, string &trained_file,
                       YAML::Node &config, string &video, string &save_dir,
                       bool x_folder, bool x_show) {

  vector<int> sliding_window_width;
  vector<int> sliding_window_height;
  vector<float> rpn_thres;
  float main_rpn_thres;
  float min_thres;
  float nms_thres;

  parse_vector<int>(config["ANCHOR_GENERATOR"]["SLIDING_WINDOW_WIDTH"],
                    sliding_window_width);
  parse_vector<int>(config["ANCHOR_GENERATOR"]["SLIDING_WINDOW_HEIGHT"],
                    sliding_window_height);
  main_rpn_thres = config["TEST"]["THRESH"].as<float>();
  min_thres = 0.0;
  nms_thres = config["TEST"]["NMS"].as<float>();
  for (int i = 0; i < sliding_window_width.size(); i++) {
    rpn_thres.push_back(main_rpn_thres);
  }

  bool x_save = true;
  if (save_dir.compare("") == 0)
    x_save = false;

  cout << "opening " << video << "..." << endl;
  vector<string> imagelist;
  vector<string>::iterator iter;
  cv::VideoCapture capture;

  if (x_folder) {
    createImgsList(imagelist, video);
    iter = imagelist.begin();
  } else {
    if (video.compare("/dev/video0") == 0)
      capture.open(0);
    else if (video.compare("/dev/video1") == 0) {
      capture.open(1);
      capture.set(3, 1920);
      capture.set(4, 1080);
    } else
      capture.open(video);

    if (!capture.isOpened()) {
      cout << "Cannot open " << video << endl;
      return;
    }
  }

  cout << "loading model..." << endl;
  RPN_detector rpn_det(model_file, trained_file, true, 1, 0, config);

  if (x_show) {
    cv::namedWindow("Cam", CV_WINDOW_NORMAL);
  }

  bool stop = false;

  int frame_count = 1;
  struct timeval start, end;

  double sum_time_forward = 0.0, aver_time_forward = 0.0;
  double sum_time_nms = 0.0, aver_time_nms = 0.0;
  double sum_time_show = 0.0, aver_time_show = 0.0;

  const string imgs_path = save_dir + "/imgs";
  const string preds_path = save_dir + "/preds";

  mkdir(save_dir.c_str(), 0755);
  mkdir(imgs_path.c_str(), 0755);
  mkdir(preds_path.c_str(), 0755);

  while (!stop) {
    cv::Mat frame;
	if (x_folder){
		if (iter==imagelist.end())
			break;
		FileStorage fs(*iter, FileStorage::READ);
		fs["Image"] >> frame;
		fs.release();
	}else if (!capture.read(frame))
      break;

    if (frame.empty()) {
      cout << "Wrong Image" << endl;
      iter++;
      continue;
    }

    int output_width = frame.cols;
    int output_height = frame.rows;
    cout << output_width << " " << output_height << endl;

    int batch_size = 0, num_channels = 0, resize_width = 0, resize_height = 0;
    rpn_det.get_input_size(batch_size, num_channels, resize_height,
                           resize_width);
    resize_width = (int)output_width * resize_height / output_height;

    cout << "input size: (" << resize_height << ", " << resize_width << ")"
         << endl;
    float enlarge_ratioh = output_height * 1.0 / resize_height,
          enlarge_ratiow = output_width * 1.0 / resize_width;
    if (frame_count == 1)
      cout << "output size: (" << output_height << ", " << output_width << ")"
           << endl;

    Mat img = frame.clone();
    Mat norm_img;
    resize(img, norm_img, Size(resize_width, resize_height));

    vector<Mat> images;
    images.push_back(norm_img);

    gettimeofday(&start, NULL);
    vector<boost::shared_ptr<Blob<float>>> outputs = rpn_det.forward(images);

    gettimeofday(&end, NULL);
    cout << "raw forward time: "
         << double(end.tv_sec - start.tv_sec) * 1000.0 +
                double(end.tv_usec - start.tv_usec) / 1.0e3
         << endl;
    sum_time_forward += double(end.tv_sec - start.tv_sec) +
                        double(end.tv_usec - start.tv_usec) / 1.0e6;
    aver_time_forward = sum_time_forward / double(frame_count);

    gettimeofday(&start, NULL);

    vector<struct Bbox> result = rpn_det.get_detection(
        images, outputs, sliding_window_width, sliding_window_height, rpn_thres,
        nms_thres, enlarge_ratiow, enlarge_ratioh);

    gettimeofday(&end, NULL);
    sum_time_nms += double(end.tv_sec - start.tv_sec) +
                    double(end.tv_usec - start.tv_usec) / 1.0e6;
    aver_time_nms = sum_time_nms / double(frame_count);

    gettimeofday(&start, NULL);

    if (x_show || x_save) {
      char str_info[100];

      cv::Mat frame_tag = frame.clone();
      for (int bbox_id = 0; bbox_id < result.size(); bbox_id++) {
        rectangle(frame_tag, result[bbox_id].rect, Scalar(0, 255, 0),
                  output_height / 400.0);
        sprintf(str_info, "%.3f", result[bbox_id].confidence);
        string prob_info(str_info);
        putText(frame_tag, prob_info,
                Point(result[bbox_id].rect.x, result[bbox_id].rect.y),
                CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255));
        cv::addWeighted(
            frame,
            1 - (result[bbox_id].confidence - min_thres) / (1.0 - min_thres),
            frame_tag,
            (result[bbox_id].confidence - min_thres) / (1.0 - min_thres), 0,
            frame);
      }

      if (x_show) {
        cv::imshow("Cam", frame);
        cv::waitKey(0);
      }
    }

    gettimeofday(&end, NULL);
    sum_time_show += double(end.tv_sec - start.tv_sec) +
                     double(end.tv_usec - start.tv_usec) / 1.0e6;
    aver_time_show = sum_time_show / double(frame_count);

    if (frame_count % 1 == 0)
      cout << "Frame." << frame_count << " "
           << "forward = " << aver_time_forward * 1000.0 << " ms, "
           << "nms = " << aver_time_nms * 1000.0 << " ms, "
           << "show = " << aver_time_show * 1000.0 << " ms" << endl;

    if (x_save) {
      char tmp[100];
      sprintf(tmp, "%04d.jpg", frame_count);
      string picPath;
      if (x_folder) {
        picPath = *iter;
      } else {
        picPath = tmp;
      }
      string img_name = basename(picPath);
      string img_path = imgs_path + "/" + img_name;
      string pred_path = preds_path + "/" + remove_ent(img_name) + ".txt";

      cv::imwrite(img_path, frame);

      ofstream outfile(pred_path.c_str());
      double x, y, width, height, score;
      for (int bbox_id = 0; bbox_id < result.size(); bbox_id++) {
        x = result[bbox_id].rect.x;
        y = result[bbox_id].rect.y;
        width = result[bbox_id].rect.width;
        height = result[bbox_id].rect.height;
        score = result[bbox_id].confidence;
        char s[100];
        sprintf(s, "%.3f", score);
        outfile << x << " " << y << " " << width << " " << height << " " << s
                << "\n";
      }
      outfile.close();
    }

    frame_count++;
    if ((cv::waitKey(x_folder ? 10 : 10) & 0xff) == 27) // Esc
      stop = true;

    if (x_folder)
      iter++;
  }
}

DEFINE_bool(folder, true, "whether src is folder");
DEFINE_bool(show, false, "whether show");
DEFINE_string(save_dir, "", "predict result save dir");

int main(int argc, char **argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("rpn prediction\n"
                          "Usage:\n"
                          "    prediction [-show] [-folder] [-save_dir] deploy "
                          "caffemodel video/img_file config_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], " ");
    return 1;
  }

  const bool is_folder = FLAGS_folder;
  const bool is_show = FLAGS_show;
  string save_dir = FLAGS_save_dir;

  string model_file = argv[1];
  string trained_file = argv[2];
  string src = argv[3];
  string config_file = argv[4];

  YAML::Node config = YAML::LoadFile(config_file);

  google::InitGoogleLogging(argv[0]);

  DetectionForVideo(model_file, trained_file, config, src, save_dir, is_folder,
                    is_show);
}
