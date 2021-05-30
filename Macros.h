#ifndef LA_MCTS_MACROS_H
#define LA_MCTS_MACROS_H

#define Assert(x,msg) \
do { \
   if(!(x)){ throw std::runtime_error(msg); } \
} while(false)

#endif //LA_MCTS_MACROS_H
