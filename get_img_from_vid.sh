# find all the mp4 files in src/data_vis/vid/ and extract the last frame to src/data_vis/img/

for f in src/data_vis/vid/*.mp4; do
  echo "Processing $f file..."

  # Extract the last frame of the MP4 file and save it as a JPG image
  output_file="${f%.*}.jpg"  # replace the file extension with jpg
  # change all 'vid' to 'img'
  output_file="${output_file//vid/img}"
  ffmpeg -i "$f" -vf 'select=gte(n\,0),setpts=N/(FRAME_RATE*TB)' -q:v 1 -vframes 1 "$output_file"

  echo "Saved $output_file"
done
