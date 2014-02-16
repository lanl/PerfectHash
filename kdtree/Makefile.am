default: libkdtree.la
all: libkdtree.la

AM_MAKEFLAGS = -j 4

DEFAULT_INCLUDES=-I. -I..

lib_LTLIBRARIES = libkdtree.la
include_HEADERS = KDTree1d.h KDTree2d.h

H_SRCS = Bounds1d.h Globals1d.h KDTree1d.h Bounds2d.h Globals2d.h KDTree2d.h

C_SRCS = Bounds1d.c KDTree1d.c Bounds2d.c KDTree2d.c

libkdtree_la_SOURCES = ${C_SRCS} ${H_SRCS}

