#!/bin/bash

cd /Users/kicirogl/workspace/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/temp_main

ls -d 201* | sort -n

arr2=($(echo ${arr[*]}| ls -d 201* | sort -n))
#cd ${arr2[${#arr2[*]}-3]}

cd 2019-05-15-20-01/

cd */

counter=0
while [ $counter -le 1 ]
do
cd ${counter}/superimposed_images

rm lift_res.mp4
rm lift_res_2.mp4
rm plot3d_flight.mp4
rm plot3d_calib.mp4
rm middle_pose.mp4
rm current_pose.mp4
rm future_pose.mp4


rm projected_res.mp4
rm global_plot.mp4
rm global_plot_calib.mp4
rm global_plot_flight.mp4

rm future_current_cov.mp4
rm potential_covs.mp4
rm potential_states.mp4
rm potential_ellipses.mp4
rm drone_traj.mp4

ffmpeg -framerate 5 -start_number 35 -i  'lift_res_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" lift_res.mp4
ffmpeg -framerate 5 -start_number 35 -i  'lift_res_2_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" lift_res_2.mp4
ffmpeg -framerate 5 -i 'global_plot_flight_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" global_plot_flight.mp4
ffmpeg -framerate 5 -i 'global_plot_calib_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih+1:x=0:y=0:color=white" global_plot_calib.mp4

ffmpeg -framerate 5 -i  'plot3d_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" current_pose.mp4
ffmpeg -framerate 5 -i  'plot3d_%01d.png' -vframes 35 -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" current_pose_calib.mp4
ffmpeg -framerate 5 -start_number 35 -i  'plot3d_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" current_pose_flight.mp4


ffmpeg -framerate 5 -i  'middle_pose_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" middle_pose.mp4
ffmpeg -framerate 5 -i 'projected_res_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" projected_res.mp4
ffmpeg -framerate 5 -i 'future_plot_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" future_pose.mp4

ffmpeg -framerate 5 -i 'potential_covs_normal_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih:x=0:y=0:color=white" potential_covs.mp4
ffmpeg -framerate 5 -i 'potential_states_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" potential_states.mp4
ffmpeg -framerate 5 -i 'potential_ellipses_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" potential_ellipses.mp4

ffmpeg -framerate 1 -i 'potential_ellipses_False_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" potential_ellipses.mp4
ffmpeg -framerate 1 -i 'potential_errors_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" potential_errors.mp4


ffmpeg -framerate 1 -i 'potential_ellipses_True_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" potential_ellipses_2.mp4
ffmpeg -framerate 2 -i 'drone_traj_2_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" drone_traj2.mp4

ffmpeg -framerate 5 -i 'plot3d_%01d.png' -vframes 15 -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih+1:x=0:y=0:color=white" plot3d_calib.mp4
ffmpeg -framerate 5 -i  'plot3d_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw+1:height=ih+1:x=0:y=0:color=white" plot3d_flight.mp4

cd ../images

rm airsim.mp4
ffmpeg -framerate 1 -vframes 15 -i 'img_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih:x=0:y=0:color=white" temp.mp4
ffmpeg -i temp.mp4 -vf hflip -c:a copy airsim.mp4
rm temp.mp4

cp airsim.mp4 ../superimposed_images

cd ../superimposed_images

ffmpeg -i potential_ellipses.mp4 -i current_pose.mp4 -filter_complex '[0:v][1:v]vstack[vid]' -map [vid] -c:v libx264 tempvid1.mp4
ffmpeg -i tempvid1.mp4 -i potential_covs.mp4 -filter_complex '[1:v]pad=height=ih+104:color=white[input2];[0:v][input2]hstack[vid]' -map [vid] -c:v libx264 tempvid2.mp4
ffmpeg -i drone_traj.mp4 -i tempvid2.mp4 -filter_complex '[1:v]pad=width=iw+160:color=white[input2];[0:v][input2]vstack[vid]' -map [vid] -c:v libx264 tempvid3.mp4
ffmpeg -i tempvid3.mp4 -i airsim.mp4 -filter_complex '[1:v]pad=width=iw+176:color=black[input2];[0:v][input2]vstack[vid]' -map [vid] -c:v libx264 output.mp4

cd ../..

((counter++))
done


