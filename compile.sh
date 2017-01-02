#!/bin/sh

cc -fPIC -c -O2 specfunc.c -o specfunc.o
#cc --shared specfunc.o -o libspecfunc.so -lgslcblas -lgsl
cc --shared specfunc.o -o libspecfunc.so
