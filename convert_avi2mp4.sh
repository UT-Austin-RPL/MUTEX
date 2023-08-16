# find all avi files in src/data_vis/vid and convert them to mp4 and save them in src/data_vis/vid with mp4 extension

for file in src/data_vis/vid/*.avi; do
    ffmpeg -i "$file" -c:v libx264 -crf 19 -preset slow -c:a aac -b:a 192k -ac 2 -strict -2 "${file%.avi}.mp4"
done
