/* specfun_standalone.h */

#ifndef SPECFUN_STANDALONE_H
#define SPECFUN_STANDALONE_H

#include <R.h>
#include <Rmath.h>

void pgamma_derivative(
    double x,
    double a,
    double scale,
    double *deriv
);

double pgamma_1st_derivative(
    double x,
    double a,
    double scale
);

#endif
