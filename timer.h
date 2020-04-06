#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>

// A work around for Windows users instead of the POSIX <sys/time.h>, it uses the files: time.h, times.h, times.c
#if defined _WIN32 || defined _WIN64
#include "time.h" 
#else
#include <sys/time.h>	
#endif

//#if defined linux
//#include <sys/time.h>
//#endif

typedef struct
{
    struct timeval startTime;
	struct timeval endTime;
} Timer;

static void startTime(Timer* timer) 
{
        gettimeofday(&(timer->startTime), NULL);
}

static void stopTime(Timer* timer) 
{
        gettimeofday(&(timer->endTime), NULL);
}

static void printElapsedTime(Timer timer, const char* s) 
{
        float t = ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                          + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
        
		printf("%s: %f s\n", s, t);
}

#endif

