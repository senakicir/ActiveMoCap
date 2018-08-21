#!/bin/bash

cd ~/Gitstuff/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/temp_main/

ls -d 201* | sort -n

arr2=($(echo ${arr[*]}| ls -d 201* | sort -n))
cd ${arr2[${#arr2[*]}-1]}
cd t*/superimposed_images

[ -e lift_res.mp4 ] && rm lift_res.mp4
[ -e openpose.mp4 ] && rm openpose.mp4
[ -e plot3d.mp4 ] && rm plot3d.mp4
[ -e projected_res.mp4 ] && rm projected_res.mp4
[ -e projected_res.mp4 ] && rm heatmaps_scales.mp4

ffmpeg -framerate 5 -start_number 25 -i 'lift_res_%01d.png' -c:v libx264 -pix_fmt yuv420p lift_res.mp4
ffmpeg -framerate 5 -i 'openpose_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" openpose.mp4
ffmpeg -framerate 5 -i 'plot3d_%01d.png' -c:v libx264 -pix_fmt yuv420p plot3d.mp4
ffmpeg -framerate 5 -i 'projected_res_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" projected_res.mp4
ffmpeg -framerate 5 -start_number 7 -i 'heatmaps_scales_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih+1:x=0:y=0:color=white" heatmaps_scales.mp4

ffmpeg -framerate 5 -start_number 7 -i 'img_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" img.mp4
