#!/bin/bash

# for all videos src/rw_demo reduce the quality of the video by 50%
# save with a new name in the same directory replacing .mp4 with _reduced.mp4
for i in src/rw_demo/*.mp4; do
    output=$(echo $i | sed 's/.mp4/_reduced.mp4/')
    ffmpeg -n -i $i -vf scale=iw/2:-1 $output
done
