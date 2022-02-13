#!/usr/bin/gnuplot

# set terminal postscript eps size 8cm,6cm enhanced color \
#     font 'Helvetica,20' linewidth 2
# set output "nagaoka_2x2.eps"
set terminal pngcairo size 640,480 enhanced color \
    font "Latin Modern Math,12"
set output "nagaoka_2x2.png"

set pm3d map interpolate 1,1
set xlabel "{/Symbol g}"
set ylabel "U"
splot "table.dat" u 1:2:4 notitle
