
#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

static void startTime(Timer* timer) {
        gettimeofday(&(timer->startTime), NULL);
}

static void stopTime(Timer* timer) {
        gettimeofday(&(timer->endTime), NULL);
}

static void printElapsedTime(Timer timer, const char* s) {
        float t = ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                        + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
            printf("%s: %f s\n", s, t);
}

#endif

