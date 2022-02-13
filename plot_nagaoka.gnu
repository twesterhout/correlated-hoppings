#!/usr/bin/gnuplot

# set terminal postscript eps size 8cm,6cm enhanced color \
#     font 'Helvetica,20' linewidth 2
# set output "nagaoka_2x2.eps"
set terminal pngcairo size 640,480 enhanced color \
    font "Latin Modern Math,12"
# set output "nagaoka_2x2_β₁_U.png"
set output "nagaoka_2x2_β₂_U.png"

set pm3d map interpolate 1,1
set xlabel "{/Symbol b}_1"
set ylabel "U"
# splot "2x2_β₂_U.dat" u 1:2:4 notitle
splot "2x2_β₂_U.dat" u 1:2:4 notitle
