#!/usr/bin/gnuplot

load 'third_party/gnuplot-palettes/moreland.pal'

dpi = 300 ## dpi (variable)
width = 80 ## mm (variable)
height = 80 ## mm (variable)
in2mm = 25.4 # mm (fixed)
pt2mm = 0.3528 # mm (fixed)
mm2px = dpi/in2mm
ptscale = pt2mm * mm2px
round(x) = x - floor(x) < 0.5 ? floor(x) : ceil(x)
wpx = round(width * mm2px)
hpx = round(height * mm2px)

# set terminal pngcairo size wpx,hpx \
#     fontscale ptscale linewidth ptscale pointscale ptscale \
#     transparent enhanced color \
#     font "Latin Modern Math,8"
set terminal pdfcairo size 8cm, 8cm \
    transparent enhanced color \
    font "Latin Modern Math,11"

set xtics out
set ytics out
set xrange [0:20]
set yrange [-1:0.5]
# set cbrange [0.5:2.5]
set border lt 1 lw 2 lc "black" back
unset colorbox
# set xlabel "U/t"
# set ylabel "F/t"
# splot "2x2_β₂_U.dat" u 1:2:4 notitle

# set pm3d map interpolate 0,0
# set dgrid3d 101,21 qnorm
# splot "temp.dat" u ($2/$3):($1/$3):4 notitle
# splot "data/4/total_spin.dat" u ($2/$3):($1/$3):4 notitle
# plot "data/4/total_spin.dat" u ($2):($1):4 with image pixels notitle

max(a, b) = a < b ? b : a
isequal(a, b) = abs(a - b) < (1e-8 + max(abs(a), abs(b)) * 1e-6)

# test
# plot "data/4/phases.dat" u 2:(isequal($3, 0.5) ? $1 : 1/0) with points lt 7 ps 1 lc "blue" notitle, \
#      "data/4/phases.dat" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points lt 7 ps 1 lc "red" notitle
# plot "data/5/phases.dat" u 2:(isequal($3, 0) ? $1 : 1/0) with points lt 7 ps 0.5 lc "blue" notitle, \
#      "data/5/phases.dat" u 2:(isequal($3, 1) ? $1 : 1/0) with points lt 7 ps 0.5 lc "green" notitle, \
#      "data/5/phases.dat" u 2:(isequal($3, 2) ? $1 : 1/0) with points lt 7 ps 0.5 lc "red" notitle
# plot "data/6/phases.dat" u 2:(isequal($3, 0.5) ? $1 : 1/0) \
#          with points ls 2 pt 7 ps 0.4 notitle, \
#      "data/6/phases.dat" u 2:(isequal($3, 1.5) ? $1 : 1/0) \
#         with points ls 4 pt 7 ps 0.4 notitle, \
#      "data/6/phases.dat" u 2:(isequal($3, 2.5) ? $1 : 1/0) \
#         with points ls 7 pt 7 ps 0.4 notitle

set output "assets/nagaoka_4.pdf"
plot "data/4/phases.dat" \
        u 2:(isequal($3, 0.5) ? $1 : 1/0) with points lt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0.5) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output "assets/nagaoka_5.pdf"
plot "data/5/phases.dat" \
        u 2:(isequal($3, 0) ? $1 : 1/0) with points lt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1) ? $1 : 1/0) with points ls 4 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 2) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 2) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output "assets/nagaoka_6.pdf"
plot "data/6/phases.dat" \
        u 2:(isequal($3, 0.5) ? $1 : 1/0) with points lt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0.5) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points ls 4 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output "assets/nagaoka_8.pdf"
plot "data/8/phases.dat" \
        u 2:(isequal($3, 0.5) ? $1 : 1/0) with points lt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0.5) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points ls 3 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points ls 5 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 3.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 3.5) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output "assets/nagaoka_9.pdf"
plot "data/9/phases.dat" \
        u 2:(isequal($3, 0) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1) ? $1 : 1/0) with points ls 3 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 2) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 2) ? $1 : 1/0) with points ls 5 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 3) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 3) ? $1 : 1/0) with points ls 7 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 4) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 4) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output "assets/nagaoka_10.pdf"
plot "data/10/phases.dat" \
        u 2:(isequal($3, 0.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0.5) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points ls 3 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points ls 5 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 3.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 3.5) ? $1 : 1/0) with points ls 7 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 4.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 4.5) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output "assets/nagaoka_12.pdf"
plot "data/12/phases.dat" \
        u 2:(isequal($3, 0.5) ? $1 : 1/0) with points lt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0.5) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1.5) ? $1 : 1/0) with points ls 2 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 2.5) ? $1 : 1/0) with points ls 4 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 3.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 3.5) ? $1 : 1/0) with points ls 5 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 4.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 4.5) ? $1 : 1/0) with points ls 7 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 5.5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 5.5) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output "assets/nagaoka_13.pdf"
plot "data/13/phases.dat" \
        u 2:(isequal($3, 0) ? $1 : 1/0) with points lt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 0) ? $1 : 1/0) with points ls 1 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 1) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 1) ? $1 : 1/0) with points ls 2 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 2) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 2) ? $1 : 1/0) with points ls 4 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 3) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 3) ? $1 : 1/0) with points ls 5 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 4) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 4) ? $1 : 1/0) with points ls 6 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 5) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 5) ? $1 : 1/0) with points ls 7 pt 7 ps 0.35 notitle, \
     "" u 2:(isequal($3, 6) ? $1 : 1/0) with points pt 7 ps 0.5 lc "black" notitle, \
     "" u 2:(isequal($3, 6) ? $1 : 1/0) with points ls 8 pt 7 ps 0.35 notitle

set output
system("for n in 4 5 6 8 9 10 12 13; do \
          convert -density 192 assets/nagaoka_$n.pdf -quality 00 assets/nagaoka_$n.png; \
        done")
