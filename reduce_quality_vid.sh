#!/bin/bash

input_file="video/rw_vid.mov"
output_file="video/rw_vid_10.mov"

ffmpeg -i "$input_file" -vf "scale=iw*0.1:-1" -c:v libx264 -crf 30 "$output_file"
