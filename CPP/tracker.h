//
// Created by lixin on 2021/5/8.
//

#ifndef LA_MCTS_TRACKER_H
#define LA_MCTS_TRACKER_H

#include <limits>
#include <string>
#include <vector>

class Tracker{
    unsigned counter   = 0;
    std::vector<double> results;
    double curt_best = std::numeric_limits<double>::max();
    std::string foldername;
public:
    Tracker()= default;
    Tracker(const std::string& name);

    void dump_trace() const;
    void track(const double& result);
};
#endif //LA_MCTS_TRACKER_H
