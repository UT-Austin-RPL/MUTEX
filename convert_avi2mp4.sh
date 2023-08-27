# find all avi files in src/data_vis/vid and convert them to mp4 and save them in src/data_vis/vid with mp4 extension

for file in src/data_vis/robot/*.avi; do
    echo "$file"
    output="${file%.avi}.mp4"
    #echo "$output"
    ffmpeg -y -i "$file" -c:v libx264 -crf 19 -preset slow -c:a aac -b:a 192k -ac 2 "$output"
    wait
done
