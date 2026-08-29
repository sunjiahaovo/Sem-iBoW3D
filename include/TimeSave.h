#pragma once

#include <utility>
#include <vector>

using namespace std;


extern vector<double> processTime;

extern vector<double> CandidateSelectionTime;

extern vector<double> GeoVerificationTime;

extern vector<double> RangeCheckTime;

extern vector<double> RegistrationTime;

extern vector<double> NoLoopRegistrationTime;

extern vector<double> WithLoopRegistrationTime;

extern vector<pair<int,double>> UpdateBoWTime;

extern vector<pair<int,double>> UpdateBoWClusterTime;

extern vector<pair<int,double>> UpdateBoWUpdateDatabaseTime;

extern vector<pair<int,double>> UpdateBoWForegroundTime;
