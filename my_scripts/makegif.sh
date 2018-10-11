#!/bin/bash

cd ~/workspace/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/temp_main/

ls -d 201* | sort -n

arr2=($(echo ${arr[*]}| ls -d 201* | sort -n))
cd ${arr2[${#arr2[*]}-1]}
cd t*/superimposed_images

[ -e lift_res.mp4 ] && rm lift_res.mp4
[ -e lift_res_2.mp4 ] && rm lift_res_2.mp4
[ -e openpose.mp4 ] && rm openpose.mp4
[ -e plot3d_flight.mp4 ] && rm plot3d_flight.mp4
[ -e plot3d_calib.mp4 ] && rm plot3d_calib.mp4

[ -e projected_res.mp4 ] && rm projected_res.mp4
[ -e heatmaps_scales.mp4 ] && rm heatmaps_scales.mp4
[ -e global_plot.mp4 ] && rm global_plot.mp4
[ -e global_plot_calib.mp4 ] && rm global_plot_calib.mp4
[ -e global_plot_flight.mp4 ] && rm global_plot_flight.mp4

[ -e ellipse_flight.mp4 ] && rm ellipse_flight.mp4
[ -e ellipse_calib.mp4 ] && rm ellipse_calib.mp4
[ -e covariance_calib.mp4 ] && rm covariance_calib.mp4
[ -e covariance_flight.mp4 ] && rm covariance_flight.mp4

[ -e future_plot.mp4 ] && rm future_plot.mp4
[ -e future_current_cov.mp4 ] && rm future_current_cov.mp4
[ -e potential_covs.mp4 ] && rm potential_covs.mp4
[ -e potential_states.mp4 ] && rm potential_states.mp4
[ -e potential_ellipses.mp4] && rm potential_ellipses.mp4


ffmpeg -framerate 5 -start_number 35 -i  'lift_res_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" lift_res.mp4
ffmpeg -framerate 5 -start_number 35 -i  'lift_res_2_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" lift_res_2.mp4
ffmpeg -framerate 2 -i 'global_plot_flight_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" global_plot_flight.mp4
ffmpeg -framerate 2 -i 'global_plot_calib_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" global_plot_calib.mp4

ffmpeg -framerate 5 -i 'openpose_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" openpose.mp4
ffmpeg -framerate 5 -i 'plot3d_%01d.png' -vframes 35 -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" plot3d_calib.mp4
ffmpeg -framerate 5 -start_number 35 -i  'plot3d_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" plot3d_flight.mp4

ffmpeg -framerate 5 -i 'projected_res_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" projected_res.mp4
ffmpeg -framerate 5 -start_number 7 -i 'heatmaps_scales_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih+1:x=0:y=0:color=white" heatmaps_scales.mp4

ffmpeg -framerate 5 -start_number 7 -i 'img_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" img.mp4

ffmpeg -framerate 2 -i 'covariance%01d.png' -vframes 35 -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" covariance_calib.mp4
ffmpeg -framerate 2 -start_number 35 -i 'covariance%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" covariance_flight.mp4

ffmpeg -framerate 2 -i 'ellipse_flight_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" ellipse_flight.mp4
ffmpeg -framerate 2 -i 'ellipse_calib_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" ellipse_calib.mp4

ffmpeg -framerate 2 -start_number 35 -i 'future_plot_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" future_plot.mp4
ffmpeg -framerate 2 -start_number 35 -i 'future_current_cov_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih+1:x=0:y=0:color=white" future_current_cov.mp4
ffmpeg -framerate 2 -start_number 35 -i 'potential_covs_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih+1:x=0:y=0:color=white" potential_covs.mp4
ffmpeg -framerate 2 -start_number 35 -i 'potential_states_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" potential_states.mp4
ffmpeg -framerate 2 -start_number 35 -i 'potential_ellipses_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" potential_ellipses.mp4

