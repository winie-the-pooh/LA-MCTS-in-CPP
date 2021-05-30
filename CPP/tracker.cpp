//
// Created by lixin on 2021/5/8.
//


#if defined(_WIN32)
#include "dirent.h"
#else
#include <sys/stat.h>
#include <dirent.h>
#endif
#define MODE (S_IRWXU | S_IRWXG | S_IRWXO)

#include <sstream>
#include <fstream>
#include "tracker.h"
#include "Macros.h"

Tracker::Tracker(const std::string& name):foldername(name){
    if( opendir(foldername.c_str())== NULL )
    {
        int ret = mkdir(foldername.c_str(), MODE);
        Assert(ret == 0,"fail to create folder.");
    }
}


void Tracker::dump_trace() const {
    std::stringstream trace_path;
    trace_path << foldername << "/result" << results.size();
    std::ofstream outfile;
    outfile.open(trace_path.str(),std::ios::out | std::ios::app);
    outfile << "[";
    for(int i = 0;i < results.size();i++)
    {
        if(i < results.size()-1)
        {
            outfile << results[i] << ", ";
        }
        else
        {
            outfile << results[i] << "]";
        }
    }
    outfile << "\n";
    outfile.close();
}

void Tracker::track(const double &result) {
    if(result < curt_best){
        curt_best = result;
    }
    results.push_back(curt_best);
    if(results.size() % 100 == 0){
        dump_trace();
    }
}