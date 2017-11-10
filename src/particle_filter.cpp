/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#define EPS 0.0001

using namespace std;

static default_random_engine rand_eng;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 250;

	// sensor noise normal distribution
	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);

	// initialize particles with sensor noises
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x + noise_x(rand_eng);
		p.y = y + noise_y(rand_eng);
		p.theta = theta + noise_theta(rand_eng);
		p.weight = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// sensor noise normal distribution
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < EPS) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);	
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      		particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      		particles[i].theta += yaw_rate * delta_t;
		}

		particles[i].x += noise_x(rand_eng);
		particles[i].y += noise_y(rand_eng);
		particles[i].theta += noise_theta(rand_eng);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (size_t i = 0; i < observations.size(); i++) {
		LandmarkObs o = observations[i];

		// initial id of landmark is set to -1
		int landmark_id = -1;
		double min_distance = numeric_limits<double>::max();

		for (size_t j = 0; j < predicted.size(); j++) {
			LandmarkObs p = predicted[j];
			double distance = dist(o.x, o.y, p.x, p.y);
			if (distance < min_distance) {
				min_distance = distance;
				landmark_id = p.id;
			}
		}
		observations[i].id = landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	double x_denom = 2 * std_landmark[0] * std_landmark[0];
  	double y_denom = 2 * std_landmark[1] * std_landmark[1];
  		
	for (int i = 0; i < num_particles; i++) {
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		vector<LandmarkObs> predictions;
		for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			// if landmark is within sensor range, add to predictions vector
			if (fabs(landmark_x - p_x) <= sensor_range &&
				fabs(landmark_y - p_y) <= sensor_range) {
				predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
			}
		}

		// transform observations from vehicle coord to map coord
		vector<LandmarkObs> transformed_obs;
		for (size_t j = 0; j < observations.size(); j++) {
			double transformed_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
			double transformed_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      		transformed_obs.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
		}

		// call data association method 
		dataAssociation(predictions, transformed_obs);

		// weight update
		particles[i].weight = 1.0;

		for (size_t j = 0; j < transformed_obs.size(); j++) {
			double obs_x = transformed_obs[j].x;
			double obs_y = transformed_obs[j].y;

			int predicted_id = transformed_obs[j].id;
			double prd_x, prd_y;
			for (size_t k = 0; k < predictions.size(); k++) {
				if (predicted_id == predictions[k].id) {
					prd_x = predictions[k].x;
					prd_y = predictions[k].y;
				}
			}
			double x_diff = obs_x - prd_x;
      		double y_diff = obs_y - prd_y;
			double b = ((x_diff * x_diff) / x_denom) + ((y_diff * y_diff) / y_denom);
			particles[i].weight *= a * exp(-b);
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;

	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}
	double max_weight = *max_element(weights.begin(), weights.end());	

	// resampling wheel
	uniform_int_distribution<int> uniform_dist(0, num_particles - 1);
	int index = uniform_dist(rand_eng);

	uniform_real_distribution<double> uniform_beta(0.0, max_weight);
	double beta = 0.0;

	for (int i = 0; i < num_particles ; i++) {
		beta += uniform_beta(rand_eng) * 2.0;
		while(beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
