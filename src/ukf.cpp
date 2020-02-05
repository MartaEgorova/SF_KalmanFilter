#include <iostream>

#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  
  is_initialized_ = false;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
 
  time_us_ = 0;
    
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  
  // tune P
  // P_(1, 1) = 0.5;
  // P_(2, 2) = 0.5;
  // P_(3, 3) = std_a_ * std_a_;
  // P_(4, 4) = std_yawdd_ * std_yawdd_;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
	
  // set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  double w = 1 / (2 * (lambda_ + n_aug_));
  for (int i = 1; i < 2 * n_aug_ + 1; ++i)
  {
      weights_(i) = w;
  }
  
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) 
  {
    // std::cout << "[ProcessMeasurement.Init]" << std::endl; 

    // set the state with the initial location
    x_.setZero();

    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
    }
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      x_(0) = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      x_(1) = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
    }

    time_us_ = meas_package.timestamp_;
    
    is_initialized_ = true;
    return;
  }

  // std::cout << "[ProcessMeasurement.Main]" << std::endl;
  
  // compute the time elapsed between the current and previous measurement, s
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  // std::cout << "dt= " << dt << std::endl;
  time_us_ = meas_package.timestamp_;

  if (meas_package.sensor_type_ == MeasurementPackage::LASER and use_laser_)
  // if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    // std::cout << "LASER" << std::endl;
    Prediction(dt);
    UpdateLidar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR and use_radar_)
  {
    // std::cout << "RADAR" << std::endl;
    Prediction(dt);
    UpdateRadar(meas_package);
  }
 
}

void UKF::CalculateAugmentedSigmaPoints(MatrixXd* Xsig_aug)
{
  // std::cout << "[CalculateAugmentedSigmaPoints]" << std::endl; 

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  
  // create augmented mean state
  x_aug.setZero();
  x_aug.head(n_x_) = x_;

  // create augmented covariance matrix
  P_aug.setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  (*Xsig_aug).col(0) = x_aug;
  (*Xsig_aug).block(0, 1, n_aug_, n_aug_) = (sqrt(lambda_ + n_aug_) * A).colwise() + x_aug;
  (*Xsig_aug).block(0, n_aug_ + 1, n_aug_, n_aug_) = (-sqrt(lambda_ + n_aug_) * A).colwise() + x_aug;
}

void UKF::PredictSigmaPoints(double delta_t, MatrixXd* Xsig_aug)
{
  // std::cout << "[PredictSigmaPoints]" << std::endl; 

  // predict sigma points
  double delta_t2 = delta_t * delta_t;
  for (int c = 0; c < 2 * n_aug_ + 1; ++c)
  {
      // avoid division by zero
      VectorXd state_update = VectorXd::Zero(n_x_);
      if ((*Xsig_aug)(4, c) == 0)
      {
        state_update(0) = (*Xsig_aug)(2, c) * cos((*Xsig_aug)(3, c)) * delta_t;
        state_update(1) = (*Xsig_aug)(2, c) * sin((*Xsig_aug)(3, c)) * delta_t;
      }
      else
      {
          double angle = (*Xsig_aug)(3, c) + (*Xsig_aug)(4, c) * delta_t;
          state_update(0) = (*Xsig_aug)(2, c) / (*Xsig_aug)(4, c) * (sin(angle) - sin((*Xsig_aug)(3, c)));
          state_update(1) = (*Xsig_aug)(2, c) / (*Xsig_aug)(4, c) * (-cos(angle) + cos((*Xsig_aug)(3, c)));
          state_update(3) = (*Xsig_aug)(4, c) * delta_t;
      }
    
      // write predicted sigma points into right column
      VectorXd noise_update(n_x_);
      noise_update << delta_t2 * cos((*Xsig_aug)(3, c)) * (*Xsig_aug)(5, c) / 2,
                      delta_t2 * sin((*Xsig_aug)(3, c)) * (*Xsig_aug)(5, c) / 2,
                      delta_t * (*Xsig_aug)(5, c),
                      delta_t2 * (*Xsig_aug)(6, c) / 2,
                      delta_t * (*Xsig_aug)(6, c);
  
    VectorXd state = (*Xsig_aug).block(0, c, n_x_, 1) + state_update + noise_update;
    Xsig_pred_.col(c) = state;
  }
}

void UKF::PredictStateStats()
{
  // std::cout << "[PredictStateStats]" << std::endl; 

  // predict state mean
  x_ = (Xsig_pred_ * weights_).rowwise().sum();

  // predict state covariance matrix
  MatrixXd D = Xsig_pred_.colwise() - x_; // deviation 
  // normalize angle to [-180, 180)
  for (int i = 0; i < D.cols(); ++i)
  {
      if (D(3, i) >= M_PI or D(3, i) < -M_PI)
      {
          D(3, i) -= 2 * M_PI * std::floor((D(3, i) + M_PI) / (2 * M_PI)); 
      }
  }
  
  P_.setZero();
  for (int i = 0; i < D.cols(); ++i)
  {
      P_ += (D.col(i) * weights_(i) * D.col(i).transpose());
  }  
}
  
void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // std::cout << "[Prediction]" << std::endl; 
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  CalculateAugmentedSigmaPoints(&Xsig_aug);
  PredictSigmaPoints(delta_t, &Xsig_aug);
  // std::cout << "[Prediction] Xsig_pred_ = " << Xsig_pred_ << std::endl; 
  PredictStateStats();
  // std::cout << "[Prediction] Xsig_pred_ = " << Xsig_pred_ << std::endl; 
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::CalculatePredictedMeasurementSpaceStats(int n_z, VectorXd& z_pred, MatrixXd& S, MatrixXd& Tc)
{
  // std::cout << "[CalculatePredictedMeasurementSpaceStats]" << std::endl; 

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i = 0; i < Xsig_pred_.cols(); ++i)
  {
      Zsig(0, i) = sqrt(Xsig_pred_(0, i) * Xsig_pred_(0, i) + Xsig_pred_(1, i) * Xsig_pred_(1, i));
      Zsig(1, i) = atan2(Xsig_pred_(1, i), Xsig_pred_(0, i));
      Zsig(2, i) = (Xsig_pred_(0, i) * cos(Xsig_pred_(3, i)) * Xsig_pred_(2, i) + Xsig_pred_(1, i) * sin(Xsig_pred_(3, i)) * Xsig_pred_(2, i)) / Zsig(0, i);
  }
  
  // calculate mean predicted measurement
  z_pred = (Zsig * weights_).rowwise().sum();
  
  // calculate innovation covariance matrix S
  MatrixXd Dz = Zsig.colwise() - z_pred; // deviation 
  // normalize angle to [-180, 180)
  for (int i = 0; i < Dz.cols(); ++i)
  {
      if (Dz(1, i) >= M_PI or Dz(1, i) < -M_PI)
      {
          Dz(1, i) -= 2 * M_PI * std::floor((Dz(1, i) + M_PI) / (2 * M_PI)); 
      }
  }
  
  S.setZero();
  for (int i = 0; i < Dz.cols(); ++i)
  {
      S += (Dz.col(i) * weights_(i) * (Dz.col(i)).transpose());
  }
  
  // form noise matrix
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;
  
  S += R;
  
  // calculate matrix for cross correlation Tc
  MatrixXd Dx = Xsig_pred_.colwise() - x_; // deviation 
  // normalize angle to [-180, 180)
  for (int i = 0; i < Dx.cols(); ++i)
  {
      if (Dx(3, i) >= M_PI or Dx(3, i) < -M_PI)
      {
          Dx(3, i) -= 2 * M_PI * std::floor((Dx(3, i) + M_PI) / (2 * M_PI)); 
      }
  }
  
  Tc.setZero();
  for (int i = 0; i < Dx.cols(); ++i)
  {
      Tc += (Dx.col(i) * weights_(i) * (Dz.col(i)).transpose());
  }
}

void UKF::PerformStateUpdate(MatrixXd& Tc, MatrixXd& S, VectorXd& z_pred, VectorXd& z)
{
  // std::cout << "[PerformStateUpdate]" << std::endl; 
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  x_ += K * (z - z_pred);
  P_ -= K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // measurement dimension, radar can measure r, phi, and r_dot
  // std::cout << "[UpdateRadar]" << std::endl; 
  int n_z = 3;
 
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  CalculatePredictedMeasurementSpaceStats(n_z, z_pred, S, Tc);
  // std::cout << "z_pred = " << z_pred << std::endl; 
  // std::cout << "S = " << S << std::endl;
  // std::cout << "z = " << meas_package.raw_measurements_ << std::endl;
  PerformStateUpdate(Tc, S, z_pred, meas_package.raw_measurements_);
  // std::cout << "[UpdateRadar] Xsig_pred_ = " << Xsig_pred_ << std::endl; 
}