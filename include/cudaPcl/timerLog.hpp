/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <cudaPcl/timer.hpp>

using std::string;
using std::ofstream;
using std::vector;

class TimerLog : public Timer
{
public:
  TimerLog(string path, uint32_t N, uint32_t it0 = 0, string name="Timer0") 
    :  Timer(), path_(path), name_(name), it0_(it0),
    fout_(path_.data(),ofstream::out), 
    t0s_(N), dts_(N,0),tSums_(N,0), tSquareSums_(N,0), Ns_(N,0) 
  {
    tic();
  };
  virtual ~TimerLog() {fout_.close();};

  virtual void tic(int32_t id=-1)
  {
    timeval t0 = this->getTimeOfDay();
    if(id < 0){
      for(int32_t i=0; i<static_cast<int32_t>(t0s_.size()); ++i) t0s_[i] = t0;
    } else if( 0<= id && id < static_cast<int32_t>(t0s_.size()))
      t0s_[id] = t0;
    else{  // add a new timer
      t0s_.push_back(t0);
      dts_.push_back(0.);
      tSums_.push_back(0.);
      tSquareSums_.push_back(0.);
      Ns_.push_back(0.);
    }
  };

  virtual float toc(int32_t id=-1)
  {
    timeval tE = this->getTimeOfDay();
    if(id < 0)
    {
      for(int32_t i=0; i<static_cast<int32_t>(dts_.size()); ++i) 
      {
        Ns_[i] ++;
        dts_[i] = this->getDtMs(t0s_[i],tE);
        if(Ns_[i] == it0_)
        { // reset sums
          tSums_[i] = 0.;
          tSquareSums_[i] = 0.;
        }else{
          tSums_[i] += dts_[i];
          tSquareSums_[i] += dts_[i]*dts_[i];
        }
      }
    } else if( 0<= id && id < static_cast<int32_t>(t0s_.size()))
    {
        Ns_[id] ++;
        dts_[id] = this->getDtMs(t0s_[id],tE);
        if(Ns_[id] == it0_)
        { // reset sums
          tSums_[id] = 0.;
          tSquareSums_[id] = 0.;
        }else{
          tSums_[id] += dts_[id];
          tSquareSums_[id] += dts_[id]*dts_[id];
        }
        return dts_[id];
    }
      return -1.;
  };

  virtual void toctic(int32_t id0, int32_t id1)
  {
    toc(id0);tic(id1);
  };

  virtual void logCycle()
  {
    if(!startLogging()) return;
    for(uint32_t i=0; i<dts_.size()-1; ++i) 
    {
      fout_<<dts_[i]<<" ";
    }
    fout_<<dts_[dts_.size()-1]<<endl;
    fout_.flush();
  };

  virtual void printStats()
  {
    if(!startLogging()) return;
    cout<<name_<<": stats over timer cycles (mean +- 3*std):\t";
    std::cout.precision(2);
    double meanTotal =0.;
    double varTotal =0.;
    for(int32_t i=0; i<static_cast<int32_t>(dts_.size()); ++i) 
    {
      double mean = tSums_[i]/(Ns_[i]-it0_);
      double var = tSquareSums_[i]/(Ns_[i]-it0_) - mean*mean;
      meanTotal += mean;
      varTotal += var;
      cout<<mean<<" +- "<<3.*sqrt(var)<<"\t";
    } 
    cout<<endl<<" => total/cycle: "<<meanTotal<<" +- "<<3.*sqrt(varTotal)<<endl;
  };

  virtual bool startLogging() const { 
    bool start = false;
    for(int32_t i=0; i<static_cast<int32_t>(dts_.size()); ++i)
      start |= Ns_[i] >= it0_;
    return start;
  };

private:
  string path_;
  string name_;
  ofstream fout_;

  uint32_t it0_; // iteration after which we start statistics

  vector<timeval> t0s_; // starts for all timings
  vector<double> dts_; // dts
  vector<double> tSums_; // sum over the time
  vector<double> tSquareSums_; // sum over squares over the time
  vector<double> Ns_; // counts of observations for each dt
};

