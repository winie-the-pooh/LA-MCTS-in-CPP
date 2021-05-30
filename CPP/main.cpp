#if defined(_WIN32)
/** This file is part of the Mingw32 package.
     *  unistd.h maps     (roughly) to io.h
     */
    #ifndef _UNISTD_H
        #define _UNISTD_H
        #include <io.h>
        #include <process.h>
    #endif /* _UNISTD_H */
#else
#include<unistd.h>
#endif

#include <iostream>
#include "LA-MCTS.h"
#include "functions.h"
int main(int argc,char* argv[])
{
    Ackley a(10);
    auto mcts = MCTS(a.getlb(),a.getub(),a.getdims(),a.getninits(),&a,a.getCp(),a.getleaf_size(),a.getkernel_type(),a.getgamma_type());
    mcts.search(100);
}