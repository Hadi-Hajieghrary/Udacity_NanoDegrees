//
//  KalmanFilter.cpp
//  Lesson 5 - Lidar and Radar Fusion With Kalman Filter
//
//  Created by Hadi Hajieghrary on 2/28/19.
//  Copyright © 2019 Hadi Hajieghrary. All rights reserved.
//

#include "KalmanFilter.hpp"
#include <iostream>

using namespace std;

DynamicModel::DynamicModel(const MatrixXd& A, const MatrixXd& B, const MatrixXd& C, const VectorXd& X0):_A(A), _B(B), _C(C), _X(X0){
    if ( A.rows() != A.cols() ){
        throw "Error: Invalid State Transition Matrix A!";
    }
    if (A.rows() != B.rows()){
        throw "Error: Invalid Control Matrix!";
    }
    if (C.cols() != A.cols()){
        throw "Error: Invalid Observation Matrix!";
    }
}

DynamicModel::DynamicModel(DynamicModel& DP):_A(DP._A),_B(DP._B), _C(DP._C){
}

DynamicModel DynamicModel::operator = (const DynamicModel& DP) {
    this->_A = DP._A;
    this->_B = DP._B;
    this->_C = DP._C;
    this->_X = DP._X;
    return *this;
}

void DynamicModel::Update(const VectorXd& U){
    _U = U;
    _X = _A * _X + _B * _U;
    _Y = _C * _X;

}



KalmanFilter::KalmanFilter(DynamicModel& DP, const MatrixXd& Q, const MatrixXd& R){
    _X = DP._X;
    _P = MatrixXd::Identity(DP._X.rows(), DP._X.rows())*1000;
    _DP = DP;
    _Q = Q;
    _R = R;
}

KalmanFilter::KalmanFilter(DynamicModel& DP, const MatrixXd& Q, const MatrixXd& R, const MatrixXd& P){
    _X = DP._X;
    _P = P;
    _DP = DP;
    _Q = Q;
    _R = R;
}

void KalmanFilter::Update(const VectorXd& U, const VectorXd& Y){

    _DP.Update(U);
    
    
    VectorXd z = VectorXd(_DP._Y.rows());
    z = _DP._Y - Y;
    MatrixXd S = _DP._C * _P * _DP._C.transpose();
    MatrixXd Sinv = S.inverse();
    
    MatrixXd Pp = _DP._A * _P * _DP._A.transpose()+ _Q;
    
    _K = Pp * _DP._C.transpose() * Sinv;
    _P = (MatrixXd::Identity(_P.rows(),_P.cols()) - _K * _DP._C)+ _R;
    _X = _X + _K * z;

}

