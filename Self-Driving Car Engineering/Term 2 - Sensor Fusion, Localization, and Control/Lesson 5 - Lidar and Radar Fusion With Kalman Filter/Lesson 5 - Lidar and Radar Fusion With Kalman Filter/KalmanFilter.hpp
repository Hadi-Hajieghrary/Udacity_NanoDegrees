//
//  KalmanFilter.hpp
//  Lesson 5 - Lidar and Radar Fusion With Kalman Filter
//
//  Created by Hadi Hajieghrary on 2/28/19.
//  Copyright © 2019 Hadi Hajieghrary. All rights reserved.
//

#ifndef KalmanFilter_hpp
#define KalmanFilter_hpp

#include <stdio.h>
#include "Eigen/Dense"
#include "Probability.hpp"

using namespace std;
using namespace Eigen;


class DynamicModel{
    
private:
    
public:
    //State Vector
    VectorXd _X;
    //State Transixion Matrix
    MatrixXd _A;
    //Input (Control) Matrix
    MatrixXd _B;
    //Control Input
    VectorXd _U;

    //Output (Measurment/Observation)
    VectorXd _Y;
    //Output (Measurment/Observation) Matrix
    MatrixXd _C;

    
    DynamicModel(){};
    DynamicModel(const MatrixXd& A, const MatrixXd& B, const MatrixXd& C, const VectorXd& X0);
    DynamicModel(DynamicModel& DP);
    DynamicModel operator = (const DynamicModel& DP);
    void Update(const VectorXd& U);
};





class KalmanFilter{
private:
    //Kalman Gain
    MatrixXd _K;
    DynamicModel _DP;
    MatrixXd _Q;
    MatrixXd _R;
    KalmanFilter(){};
    
public:
    // Estimated State
    VectorXd _X;
    //State Estimation Covarience Matrix
    MatrixXd _P;
    
    KalmanFilter(DynamicModel& DP, const MatrixXd& Q, const MatrixXd& R);
    KalmanFilter(DynamicModel& DP, const MatrixXd& Q, const MatrixXd& R, const MatrixXd& P);
    
    void Update(const VectorXd& U, const VectorXd& Y);
    
};



#endif /* KalmanFilter_hpp */

