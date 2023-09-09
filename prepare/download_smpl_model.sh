mkdir -p deps/
cd deps/

echo "The smpl model will be stored in the './deps' folder"

# SMPL Models
echo "Downloading"
gdown "https://drive.google.com/uc?id=1qrFkPZyRwRGd0Q3EY76K8oJaIgs_WK9i"
echo "Extracting"
tar xfzv smpl.tar.gz
echo "Cleaning"
rm smpl.tar.gz

echo "Downloading done!"
