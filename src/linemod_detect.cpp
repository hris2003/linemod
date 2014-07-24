/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <ecto/ecto.hpp>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <map>

#include <boost/foreach.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <opencv2/rgbd/rgbd.hpp>
#include "point_cloud2_proxy.h"
#include <sensor_msgs/PointCloud2.h>

#include <object_recognition_core/common/json.hpp>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/document.h>
#include <object_recognition_core/db/model_utils.h>

#include <object_recognition_core/db/ModelReader.h>
#include <object_recognition_core/common/pose_result.h>

#define USE_GLUT 1
#if USE_GLUT
#include <object_recognition_renderer/renderer_glut.h>
#else
#include <object_recognition_renderer/renderer_osmesa.h>
#endif
#include <object_recognition_renderer/utils.h>

#include "db_linemod.h"
#include <math.h>

using ecto::tendrils;
using ecto::spore;
using object_recognition_core::db::ObjectId;
using object_recognition_core::common::PoseResult;
using object_recognition_core::db::ObjectDbPtr;

void drawResponse(const std::vector<cv::linemod::Template>& templates,
		int num_modalities, cv::Mat& dst, cv::Point offset, int T) {
	static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255), CV_RGB(0, 255, 0),
			CV_RGB(255, 255, 0), CV_RGB(255, 140, 0), CV_RGB(255, 0, 0) };
	cv::Point p(offset.x + 20, offset.y + 20);
	cv::circle(dst, p, T / 2, COLORS[4]);
	for (int m = 0; m < num_modalities; ++m) {
// NOTE: Original demo recalculated max response for each feature in the TxT
// box around it and chose the display color based on that response. Here
// the display color just depends on the modality.
		cv::Scalar color = COLORS[m];

		for (int i = 0; i < (int) templates[m].features.size(); ++i) {
			cv::linemod::Feature f = templates[m].features[i];
			cv::Point pt(f.x + offset.x, f.y + offset.y);
			cv::circle(dst, pt, T / 2, color);
		}
	}
}

//========================================================================
/**
 * Can we just base on the templates to get the right position of the origin point
 */
void get3DOriginPoint(cv::Mat_<cv::Vec3f>& cloudImage, cv::Vec3f& pt) {

	//cv::linemod::Feature f = templates[1].features[0];
	//int x = -1, y = -1;
	float d = 1e+5, min_x = 1e+5, max_y = -1e+5; //cloudImage( f.y + offset.y, f.x + offset.x)[2];
	//cv::Vec3f d_origin(d,d,d);
	//we need the min_x, max_y and d_min

	for (int x = 0; x < cloudImage.rows; x++)
		for (int y = 0; y < cloudImage.cols; y++) {
			//f = templates[0].features[i];
			if (!isnan(cloudImage(x, y)[0]))//is it faster this way? Not sure

				if (min_x > cloudImage(x, y)[0]) {
					min_x = cloudImage(x, y)[0];
					//x = f.x + offset.x, y = f.y + offset.y;
				}

			if (!isnan(cloudImage(x, y)[1]))

				if (max_y < cloudImage(x, y)[1]) {
					max_y = cloudImage(x, y)[1];
					//x = f.x + offset.x, y = f.y + offset.y;
				}
			if (!isnan(cloudImage(x, y)[2]))

				if (d > cloudImage(x, y)[2]) {
					d = cloudImage(x, y)[2];
					//x = f.x + offset.x, y = f.y + offset.y;
				}
		}
	pt(0) = min_x;
	pt(1) = max_y;
	pt(2) = d;
}

void getBoundingbox(const std::vector<cv::linemod::Template>& templates,
		cv::Mat_<cv::Vec3f>& cloudImage, cv::Point offset,
		cv::Rect_<int>& rect) {

	cv::linemod::Feature f = templates[0].features[0];
	int x_min = 10000, y_min = 10000, x_max = -1, y_max = -1;

	for (int i = 0; i < (int) templates[0].features.size(); ++i) {
		f = templates[0].features[i];
		if (f.x < x_min)
			x_min = f.x;
		if (f.y < y_min)
			y_min = f.y;
		if (f.x > x_max)
			x_max = f.x;
		if (f.y > y_max)
			y_max = f.y;
	}
	rect.x = x_min + offset.x;
	rect.y = y_min + offset.y;
	rect.width = x_max - x_min;
	rect.height = y_max - y_min;
}

void getPointsInRect(cv::Mat_<cv::Vec3f> src_cloud,
		cv::Mat_<cv::Vec3f> dest_cloud, std::vector<cv::Vec3f>& s_neighbors,
		std::vector<cv::Vec3f>& d_neighbors, cv::Rect rect) {
	s_neighbors.clear();
	d_neighbors.clear();
	for (int i = rect.x; (i < rect.x + rect.width && i < src_cloud.cols); ++i)
		for (int j = rect.y; (j < rect.y + rect.height && j < src_cloud.rows);
				++j) {
			if (!cv::checkRange(dest_cloud(j, i)))
				continue;
			if (!cv::checkRange(src_cloud(j, i)))
				continue;
			cv::Vec3f s;
			s(0) = src_cloud(j, i)[0];
			s(1) = src_cloud(j, i)[1];
			s(2) = src_cloud(j, i)[2];
			cv::Vec3f d;
			d(0) = dest_cloud(j, i)[0];
			d(1) = dest_cloud(j, i)[1];
			d(2) = dest_cloud(j, i)[2];
			s_neighbors.push_back(s);
			d_neighbors.push_back(d);

		}

}

/**
 * @brief try to get the source (color image and depth image) generated by view_generator during training
 * phase. Known input are the matrix R and T for the rendering, and a vector of cv::Mat to store the output
 * images.
 */
//void getSourceFromTR(cv::Mat T_in, cv::Mat R_in, std::vector<cv::Mat>& rendered_images, std::string object_id){
void setupRenderer(std::string object_id,
		std::map<std::string, RendererGlut>& ri_map,
		std::vector<cv::Mat>& imgs_ref, std::vector<cv::Mat>& mask_ref) {
//Inspired from linemod_train.cpp

// Steps are:
	//1. get the mesh of the object_id
	//2. Initiate the Renderer and RendererIterator with right same input at linemod_train.cpp
	//3. Iterate the Renderer until finding the matched T & R of T_in & R_in
	//4. Save the rendered images to the appropriate output

	//ACTION!!!!

	object_recognition_core::db::ObjectDbParameters db_params;
	db_params.set_parameter("type", "CouchDB");
	db_params.set_parameter("root", "http://localhost:5984");
	db_params.set_parameter("collection", "object_recognition");

	// Get the document for the object_id_ from the DB
	object_recognition_core::db::ObjectDbPtr db = db_params.generateDb();
	object_recognition_core::db::Documents documents =
			object_recognition_core::db::ModelDocuments(db,
					std::vector<object_recognition_core::db::ObjectId>(1,
							object_id), "mesh");
	if (documents.empty()) {
		std::cerr << "Skipping object id \"" << object_id
				<< "\" : no mesh in the DB" << std::endl;
		return;
	}

	// Get the list of _attachments and figure out the original one
	object_recognition_core::db::Document document = documents[0];
	std::vector<std::string> attachments_names = document.attachment_names();
	std::string mesh_path;
	BOOST_FOREACH(const std::string& attachment_name, attachments_names){
	if (attachment_name.find("original") != 0)
	continue;
	// Create a temporary file
	char mesh_path_tmp[L_tmpnam];
	mkstemp(mesh_path_tmp);
	mesh_path = std::string(mesh_path_tmp) + attachment_name.substr(8);

	// Load the mesh and save it to the temporary file
	std::ofstream mesh_file;
	mesh_file.open(mesh_path.c_str());
	document.get_attachment_stream(attachment_name, mesh_file);
	mesh_file.close();
}

// Define the display
	size_t width = 640, height = 480;
	double near = 0.1, far = 1000;
	double focal_length_x = 525, focal_length_y = 525;

// the model name can be specified on the command line.
#if USE_GLUT
	RendererGlut renderer = RendererGlut(mesh_path);
#else
	RendererOSMesa renderer = RendererOSMesa(mesh_path);
#endif
	renderer.set_parameters(width, height, focal_length_x, focal_length_y, near,
			far);
	std::remove(mesh_path.c_str());

	RendererIterator renderer_iterator = RendererIterator(&renderer, 150);

	cv::Mat image, depth, mask;
	cv::Mat_<unsigned short> depth_short;
	cv::Matx33d R;
	cv::Vec3d T;
	for (size_t i = 0; !renderer_iterator.isDone(); ++i, ++renderer_iterator) {

		std::stringstream status;
//status << "Loading images " << (i+1) << "/"
//    << renderer_iterator.n_templates();
//std::cout << status.str();

		renderer_iterator.render(image, depth, mask);
		R = renderer_iterator.R();
		T = renderer_iterator.T();

		//depth.convertTo(depth_short, CV_16U, 1000);
		depth.convertTo(depth, CV_32F, 0.001);  //convert to meter
		imgs_ref.push_back(depth);
		mask_ref.push_back(mask);
	}
//RendererIterator renderer_iterator = RendererIterator(&renderer, 150);
// the model name can be specified on the command line.
//ri_map.insert(std::pair<std::string, RendererGlut> (object_id, renderer));

	std::cout << "Here finally with a RendererIterator defined! " << std::endl;
}

/**
 * Perform a 3D affine transformation on the src cloud
 */
void transformPoints(cv::Mat_<cv::Vec3f> src, cv::Mat_<cv::Vec3f>& dest,
		cv::Matx33f R, cv::Vec3f T) {
//Transformation R, T: dest_i = R * src_i + T
	//cv::Mat_<cv::Vec3f> dest_tmp (src);
	for (int i = 0; i < src.cols; i++)
		for (int j = 0; j < src.rows; j++) {
			dest(j, i) = R * src(j, i) + T;
		}
}
/**
 * Calculate euclidean distance between two set of 3D point.
 * This supposes that the size of the two point clouds are the same. TO BEWARE!
 */
double getDistance2Clouds(cv::Mat_<cv::Vec3f> cloud1,
		cv::Mat_<cv::Vec3f> cloud2) {
	double d = 0;
	double min = 100000;
	double max = 0;
	/*
	 for (int i = 0; i < cloud1.rows; i++)
	 for (int j = 0; j < cloud1.cols; j++) {
	 */
	for (int i = cloud1.rows / 4; i < 3 * cloud1.rows / 4; i++)
		for (int j = cloud1.cols / 4; j < 3 * cloud1.cols / 4; j++) {
			if (!cv::checkRange(cloud1(i, j)))
				continue;
			if (!cv::checkRange(cloud2(i, j)))
				continue;
			double dt = sqrt(
					pow((cloud1(i, j)[0] - cloud2(i, j)[0]), 2)
							+ pow(cloud1(i, j)[1] - cloud2(i, j)[1], 2)
							+ pow(cloud1(i, j)[2] - cloud2(i, j)[2], 2));
			d += dt;
			if (dt < min)
				min = dt;
			if (dt > max)
				max = dt;

		}
	//std::cout<<"Distance: \tmin:"<<min<<"\tmax:"<<max<<"\n";

	return 9 * d / (cloud1.rows * cloud1.cols);
}

void getMean(std::vector<cv::Vec3f> pts, cv::Vec3f& centroid) {
	centroid(0) = 0;
	centroid(1) = 0;
	centroid(2) = 0;
	for (int i = 0; i < pts.size(); i++) {
		if (!cv::checkRange(pts.at(i)))
			continue;

		centroid(0) += pts.at(i)(0);
		centroid(1) += pts.at(i)(1);
		centroid(2) += pts.at(i)(2);
	}
	centroid(0) /= pts.size();
	centroid(1) /= pts.size();
	centroid(2) /= pts.size();

}
/**
 * Check the ratio of in_cloud points that have the expected depth.
 * The input Mask is there to indicate inliers.
 * Threshold is distance threshold between input point and expected depth point
 */
float getExpectedInliersRatio(cv::Mat_<cv::Vec3f> in_cloud,
		cv::Mat_<cv::Vec3f> expected_depth, cv::Mat mask, double threshold) {
	int count = 0;
	int max_count = 0;
	for (int i = 0; i < in_cloud.cols; i++)
		for (int j = 0; j < in_cloud.rows; j++) {
			if (mask.at<int>(j, i) == 0)
				continue;
			if (!cv::checkRange(in_cloud(j, i)))
				continue;
			if (!cv::checkRange(expected_depth(j, i)))
				continue;
			double dt = sqrt(
					pow((in_cloud(j, i)[0] - expected_depth(j, i)[0]), 2)
							+ pow(in_cloud(j, i)[1] - expected_depth(j, i)[1],
									2)
							+ pow(in_cloud(j, i)[2] - expected_depth(j, i)[2],
									2));
			if (dt < threshold) {
				count++;
			}

			max_count++;
		}
	if (max_count == 0)
		return 0;

	return ((float) count) / max_count;
}
/**
 * Subtract element-wise a vector of points 3D by a point 3D
 */
void doSubtraction(std::vector<cv::Vec3f>& pts, cv::Vec3f pt) {
	//cv::Vec3f rs;
	//rs(0) = 0; rs(1) = 0; rs(2) = 0;
	for (int i = 0; i < pts.size(); i++) {
		pts.at(i)(0) = pts.at(i)(0) - pt(0);
		pts.at(i)(1) = pts.at(i)(1) - pt(1);
		pts.at(i)(2) = pts.at(i)(2) - pt(2);
	}
	return;
}
/**
 * @brief Calculating the translation and rotation needed to transform the source cloud to the destination cloud
 * @param src
 * @param dest
 * @param R1
 * @param T1
 * @param R_out
 * @param T_out
 */
void icpCloudToCloud(cv::Mat_<cv::Vec3f> src, cv::Mat_<cv::Vec3f> dest,
		cv::Matx33f& R, cv::Vec3f& T) {

	cv::Matx33f r;
	cv::Vec3f t;

	// BIBLE: http://stackoverflow.com/questions/20528094/computing-the-3d-transformation-between-two-sets-of-points
	// LOOP until min_dist or max_iter

	int n_iter = 100, it = 0, n_side = 20;
	std::vector<cv::Vec3f> s_neighbors, d_neighbors;
	cv::Mat_<cv::Vec3f> src_in(src);
	//transformPoints(src, src, R, cv::Vec3f(0,0,0));
	cv::Mat R0 = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	double min_dist = getDistance2Clouds(src_in, dest);
	/*
	 std::cout << "begin distance: " << min_dist << "\n"
	 << src(src.rows / 2, src.cols / 2) << "\n"
	 << dest(dest.rows / 2, dest.cols / 2) << "\n";
	 */
	while ((min_dist > 0.01) && (it < n_iter)) {
		it++;

		//1. Get the "random" subset of src and dest cloud
		//1.1. Generate the random location
		cv::Rect rect((src_in.cols / 3) + rand() % (src_in.cols / 4),
				(src_in.rows / 3) + rand() % (src_in.rows / 4), n_side, n_side);
		getPointsInRect(src_in, dest, s_neighbors, d_neighbors, rect);
		//std::cout<<"points (size) : "<<s_neighbors.size()<<"\n";
		if (s_neighbors.size() == 0)
			continue;

		//2. Calculate centroid of each subset
		cv::Vec3f s_centroid;
		getMean(s_neighbors, s_centroid);
		cv::Vec3f d_centroid;
		getMean(d_neighbors, d_centroid);

		doSubtraction(s_neighbors, s_centroid);
		doSubtraction(d_neighbors, d_centroid);
		//3. Calculate the covariance matrix M
		cv::Matx33f covariance;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				covariance(i, j) = 0;
				for (int l = 0; l < s_neighbors.size(); l++) {
					covariance(i, j) += s_neighbors.at(l)[i]
							* d_neighbors.at(l)[j];
				}
			}
		//4. Call SVD to get w_u_vt
		cv::Mat w, u, vt;
		cv::SVD::compute(covariance, w, u, vt);
		//std::cout<<"SVD::u \n"<<u<<std::endl;
		//std::cout<<"SVD::v \n"<<vt.t()<<std::endl;
		//5. Calculate R matrix: R = V * U.t
		r = cv::Mat(vt.t() * u.t());

		//6. Calculate T matrix: T = C_b - R * C_a
		t = d_centroid - r * s_centroid;
		//cv::subtract(d_centroid, r_c_a, t);

		//Make sure that returned result is not null
		if (!cv::checkRange(r))
			continue;
		if (!cv::checkRange(t))
			continue;
		//7. get the distance between the new cloud position and the dest_cloud
		// and update new distance if position
		cv::Mat_<cv::Vec3f> transposed_src = src_in.clone();
		transformPoints(src_in, transposed_src, r, t);
		double distance = getDistance2Clouds(transposed_src, dest);

		if (distance < min_dist) {	  //and update min_dist
			transformPoints(src_in, src_in, r, t);
			min_dist = getDistance2Clouds(src_in, dest);
			R = R * r;
			T = r * T;
			cv::add(T, t, T);

		}
	}
	s_neighbors.clear();
	d_neighbors.clear();

	min_dist = getDistance2Clouds(src_in, dest);

	//std::cout << "End distance: " << min_dist << "\n"
	//<< src_in(src_in.rows / 2, src_in.cols / 2) << "\n"
	//<< dest(dest.rows / 2, dest.cols / 2) << "\n";
//	 std::cout << "Found rotation: " << R << std::endl;

//	 std::cout << "Found translation: " << T << std::endl;

}

//=======================================================================
namespace ecto_linemod {
struct Detector: public object_recognition_core::db::bases::ModelReaderBase {
	void parameter_callback(
			const object_recognition_core::db::Documents & db_documents) {
		/*if (submethod.get_str() == "DefaultLINEMOD")
		 detector_ = cv::linemod::getDefaultLINEMOD();
		 else
		 throw std::runtime_error("Unsupported method. Supported ones are: DefaultLINEMOD");*/

		detector_ = cv::linemod::getDefaultLINEMOD();
		BOOST_FOREACH(const object_recognition_core::db::Document & document, db_documents){
		std::string object_id = document.get_field<ObjectId>("object_id");

		// Load the detector for that class
		cv::linemod::Detector detector;
		document.get_attachment<cv::linemod::Detector>("detector",
				detector);
		std::string object_id_in_db = detector.classIds()[0];
		for (size_t template_id = 0; template_id < detector.numTemplates();
				++template_id)
		detector_->addSyntheticTemplate(
				detector.getTemplates(object_id_in_db, template_id),
				object_id);

		// Deal with the poses
		document.get_attachment<std::vector<cv::Mat> >("Rs",
				Rs_[object_id]);
		document.get_attachment<std::vector<cv::Mat> >("Ts",
				Ts_[object_id]);

		std::cout << "Loaded " << object_id << std::endl;

		setupRenderer(object_id, ri_map_, images_ref_, masks_ref_);
	}
}

static void declare_params(tendrils& params) {
	object_recognition_core::db::bases::declare_params_impl(params,
			"LINEMOD");
	params.declare(&Detector::threshold_, "threshold",
			"Matching threshold, as a percentage", 93.0f);
	params.declare(&Detector::visualize_, "visualize",
			"If True, visualize the output.", false);

}

static void declare_io(const tendrils& params, tendrils& inputs,
		tendrils& outputs) {
	inputs.declare(&Detector::color_, "image", "An rgb full frame image.");
	inputs.declare(&Detector::depth_, "depth", "The 16bit depth image.");
	inputs.declare(&Detector::K_image_, "K_image", "The calibration matrix").required();
	inputs.declare(&Detector::K_depth_, "K_depth", "The calibration matrix").required();

	outputs.declare(&Detector::pose_results_, "pose_results",
			"The results of object recognition");
}

void configure(const tendrils& params, const tendrils& inputs,
		const tendrils& outputs) {
	configure_impl();
}

int process(const tendrils& inputs, const tendrils& outputs) {
	// Resize color to 640x480
	/// @todo Move resizing to separate cell, and try LINE-MOD w/ SXGA images

	cv::Mat color;
	if (color_->rows > 960)
	cv::pyrDown(color_->rowRange(0, 960), color);
	else
	color_->copyTo(color);

	pose_results_->clear();

	if (detector_->classIds().empty())
	return ecto::OK;

	std::vector<cv::Mat> sources;
	sources.push_back(color);
	sources.push_back(*depth_);

	std::vector<cv::linemod::Match> matches, filteredMatches;
	detector_->match(sources, *threshold_, matches);
	cv::Mat display = color;
	int num_modalities = (int) detector_->getModalities().size();

	//eliminate duplicated positions
	std::vector<cv::Point> posPoint;
	std::vector<int> maxSize;
	//filteredMatches->clear();
	if (matches.size() > 0) {
		filteredMatches.push_back(matches.at(0));
	}
	//========================================================
	//ICP for depth info
	cv::Mat_<cv::Vec3f> m_3DImg;
	cv::Mat_<float> K;
	K_depth_->convertTo(K, CV_32F);
	//cv::checkRange(*depth_);
	cv::depthTo3d(*depth_, K, m_3DImg);
	//==============================================================
	//get the finest icp-ed matches only

	//Matx to switch between Y and Z
	cv::Mat R_yz =
	(cv::Mat_<float>(3, 3) << -1, 0, 0, 0, 0, -1, 0, 1, 0);

	int count = 0;

	double gtf = 0;
	BOOST_FOREACH(const cv::linemod::Match & match, matches) {
		const std::vector<cv::linemod::Template>& templates =
		detector_->getTemplates(match.class_id, match.template_id);
		if (*visualize_)
		drawResponse(templates, num_modalities, display,
				cv::Point(match.x, match.y), detector_->getT(0));

		if (count > 10)
		break;
		count++;
		cv::Mat mat = cv::Mat::eye(3, 3, CV_32F);
		float* data = reinterpret_cast<float*>(mat.data);//cast mat.data to float*

		cv::Matx33f R(data), R1(data), R0(data);
		cv::Vec3f T(0, 0, 0), T0(0, 0, 0), T1(0,0,0);

		{
			cv::Mat R_in =
			Rs_.at(match.class_id)[match.template_id].clone();
			cv::Mat T_in =
			Ts_.at(match.class_id)[match.template_id].clone();
			R_in.convertTo(R, CV_32F);
			R_in.convertTo(R1, CV_32F);
			T_in.convertTo(T, CV_32F);
			T_in.convertTo(T1, CV_32F);
		}

		//======================================
		//Move the pose_result to the match position
		cv::linemod::Feature f = templates[1].features[0];

		//Get the good position by filtering the cloud in the zone of the returned match
		cv::Rect_<int> rect(0, 0, -1, -1);
		cv::Point d_point(0,0);//to save the location of the depth value of the origin point on the acquired image

		getBoundingbox(templates, m_3DImg, cv::Point(match.x, match.y),
				rect);
		cv::Vec3f T_origin(0,0,0);

		image_ref_ = images_ref_.at(match.template_id);

		rect.width = image_ref_.cols;
		rect.height = image_ref_.rows;

		depth_real = m_3DImg(rect);

		cv::Mat_<cv::Vec3f> crop_out;
		cv::depthTo3d(image_ref_, K, crop_out);
		//get3DOriginPoint(crop_out, T_origin);//does not get us anywhere :-(

		T = T0 + crop_out(rect.height / 2, rect.width / 2);
		//R = R1;
		icpCloudToCloud(crop_out, depth_real, R0, T);

		float threshold = 0.02;
		float good_to_fit = getExpectedInliersRatio(crop_out, depth_real,
				masks_ref_.at(match.template_id), threshold);

		if (good_to_fit < 0.75)
			continue;

		gtf = good_to_fit;
		//std::cout << "Good to fit is: " << good_to_fit << "\n\n";
		//T = m_3DImg(rect.y + (rect.height / 2), rect.x + (rect.width / 2));
		//T(2) = T(2) + 0.03;	//to adjust the pose_result position to the center of the cloud, not on the cloud

		//======================================

		if (!cv::checkRange(R0))
		continue;
		if (!cv::checkRange(T))
		continue;

		// Fill the Pose object
		PoseResult pose_result;
		pose_result.set_R(cv::Mat(R0.t()*R));//*cv::Matx33f(R_yz)));      //to fix rotation, how?

		pose_result.set_T(cv::Mat(T));//cv::Vec3f(T(1)+0.01*(count-1), T(2),T(3))));
		pose_result.set_object_id(db_, match.class_id);

		/*
		 if (cv::norm(T1 - cv::Vec3f(-0.09222, 0.794667, 0)) > 0.01){
		 continue;
		 }
		 std::cout << "T_in:\n" << T1 << std::endl;
		 std::cout << "T_out:\n" << T << std::endl;
		 std::cout << "R_in:\n" << R1 << std::endl;
		 */
		//=====================================================
		// Add the point cloud to the pose_result
		// Add the cluster of points
		/*
		 std::vector<sensor_msgs::PointCloud2Ptr> ros_clouds (1);
		 ros_clouds[0].reset(new sensor_msgs::PointCloud2());
		 sensor_msgs::PointCloud2Proxy<sensor_msgs::PointXYZ> proxy(*(ros_clouds[0]));
		 // Add the cloud
		 proxy.resize(templates[1].features.size());
		 sensor_msgs::PointXYZ *iter = &(proxy[0]);

		 for (int i = 0; i < (int) templates[1].features.size(); ++i, ++iter)
		 {
		 f = templates[1].features[i];
		 cv::Vec3f res = m_3DImg( f.y + match.y, f.x + match.x);
		 iter->x = res[0];
		 iter->y = res[1];
		 iter->z = res[2];

		 }
		 pose_result.set_clouds(ros_clouds);

		 */
		pose_result.set_confidence(match.similarity);
		pose_results_->push_back(pose_result);
		//=====================================================

		//break;

	};

	if (*visualize_) {
		cv::namedWindow("LINEMOD");
		cv::imshow("LINEMOD", display);
		// Display the rendered image
		/*
		 cv::namedWindow("Rendering");
		 cv::namedWindow("Depth_in");
		 if (!image_ref_.empty()) {
		 cv::imshow("Rendering", image_ref_);
		 cv::imshow("Depth_in", depth_real);
		 }
		 */
		cv::waitKey(1);
	}
	return ecto::OK;
}

/** LINE-MOD detector */
cv::Ptr<cv::linemod::Detector> detector_;
// Parameters
spore<float> threshold_;
// Inputs
spore<cv::Mat> color_, depth_;
// Calibration matrix of the camera
spore<cv::Mat> K_image_, K_depth_;
/** The DB parameters as a JSON string */
ecto::spore<std::string> json_db_;
cv::Mat image_ref_, mask_out;
cv::Mat_<cv::Vec3f> depth_real;
// A dictionary of RendererIterator for object_ids
std::map<std::string, RendererGlut> ri_map_;
std::vector<cv::Mat> images_ref_;
std::vector<cv::Mat> masks_ref_;
std::vector<cv::Mat> Rs_ref_;
std::vector<cv::Mat> Ts_ref_;
/*
 #if USE_GLUT
 //RendererGlut renderer = RendererGlut(mesh_path);
 std::map<std::string, RendererGlut> ri_map_;
 #else
 //RendererOSMesa renderer = RendererOSMesa(mesh_path);
 std::map<std::string, RendererOSMesa> ri_map_;
 #endif
 */
/** True or False to output debug image */
ecto::spore<bool> visualize_;
/** The object recognition results */
ecto::spore<std::vector<PoseResult> > pose_results_;
/** The rotations, per object and per template */
std::map<std::string, std::vector<cv::Mat> > Rs_;
/** The translations, per object and per template */
std::map<std::string, std::vector<cv::Mat> > Ts_;
};

}
		// namespace ecto_linemod

ECTO_CELL(ecto_linemod, ecto_linemod::Detector, "Detector",
"Use LINE-MOD for object detection.")
