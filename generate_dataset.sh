#!/bin/bash

# URL of the file to download
file_url="https://drive.usercontent.google.com/download?id=1Ur9Oo45-y35sh_jI-k5X0IoseTPxzdbe&export=download&confirm=t&uuid=d035aa33-39ba-4f17-87e9-2f7c3726ea06"

# Name of the downloaded file
downloaded_file="treecover_segmentation_dataset.zip"

# Download the file
echo "Downloading file from $file_url ..."
curl -L -o "$downloaded_file" "$file_url" --progress-bar

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download the file."
    exit 1
fi

# Unzip the file
echo "Unzipping $downloaded_file ..."
unzip -q $downloaded_files

# Check if unzip was successful
if [ $? -ne 0 ]; then
    echo "Failed to unzip the file."
    exit 1
fi

echo "File has been downloaded and unzipped successfully."