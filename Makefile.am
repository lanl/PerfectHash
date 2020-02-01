bin_PROGRAMS = sort sort2d remap remap2d neigh neigh2d table tablelarge

ACLOCAL_AMFLAGS = -I m4

SUBDIRS = kdtree

sort_C_SRCS = sort.c gpu.c timer.c
sort_H_SRCS = gpu.h timer.h
sort_SOURCES = ${sort_C_SRCS} ${sort_H_SRCS}

sort2d_C_SRCS = sort2d.c gpu.c timer.c
sort2d_H_SRCS = gpu.h timer.h
sort2d_SOURCES = ${sort2d_C_SRCS} ${sort2d_H_SRCS}

remap_C_SRCS = remap.c gpu.c timer.c
remap_H_SRCS = gpu.h timer.h
remap_SOURCES = ${remap_C_SRCS} ${remap_H_SRCS}
remap_LDADD = kdtree/libkdtree.la

remap2d_C_SRCS = remap2d.c gpu.c timer.c
remap2d_H_SRCS = gpu.h timer.h
remap2d_SOURCES = ${remap2d_C_SRCS} ${remap2d_H_SRCS}
remap2d_LDADD = kdtree/libkdtree.la

neigh_C_SRCS = neigh.c gpu.c timer.c
neigh_H_SRCS = gpu.h timer.h
neigh_SOURCES = ${neigh_C_SRCS} ${neigh_H_SRCS}
neigh_LDADD = kdtree/libkdtree.la

neigh2d_C_SRCS = neigh2d.c gpu.c timer.c
neigh2d_H_SRCS = gpu.h timer.h
neigh2d_SOURCES = ${neigh2d_C_SRCS} ${neigh2d_H_SRCS}
neigh2d_LDADD = kdtree/libkdtree.la

table_C_SRCS = table.c gpu.c timer.c
table_H_SRCS = gpu.h timer.h
table_SOURCES = ${table_C_SRCS} ${table_H_SRCS} table.data

tablelarge_C_SRCS = tablelarge.c gpu.c timer.c
tablelarge_H_SRCS = gpu.h timer.h
tablelarge_SOURCES = ${tablelarge_C_SRCS} ${tablelarge_H_SRCS} tablelarge.data

LDADD = -lm

EXTRA_DIST = sort_kern.cl sort2d_kern.cl remap_kern.cl remap2d_kern.cl neigh_kern.cl neigh2d_kern.cl table_kern.cl table.data tablelarge_kern.cl tablelarge.data config.h.in APACHE_LICENSE-2_0 AUTHORS README LICENSE Papers.bib NOTICE

kdtree/libkdtree.la:
	cd kdtree; make

distclean-local:
	rm -f config.h.in~
	rm -f hash-v*.tar.gz
	rm -rf build
	rm -rf autom4te.cache

