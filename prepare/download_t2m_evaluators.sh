mkdir -p deps/
cd deps/

echo "The t2m evaluators will be stored in the './deps' folder"

# HumanAct12 poses
echo "Downloading"
gdown "https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
echo "Extracting"
tar xfzv t2m.tar.gz
echo "Cleaning"
rm t2m.tar.gz

echo "Downloading done!"
