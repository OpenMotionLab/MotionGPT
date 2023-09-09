#!/bin/bash
echo "Extraction of the archives"
echo

cd deps/smplh
mkdir tmp
cd tmp

tar xfv ../smplh.tar.xz
unzip ../mano_v1_2.zip

cd ../../../
echo
echo "Done!"
echo
echo "Clean and merge models"
echo

python prepare/merge_smplh_mano.py --smplh-fn deps/smplh/tmp/male/model.npz --mano-left-fn deps/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn deps/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder deps/smplh/

python prepare/merge_smplh_mano.py --smplh-fn deps/smplh/tmp/female/model.npz --mano-left-fn deps/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn deps/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder deps/smplh/

python prepare/merge_smplh_mano.py --smplh-fn deps/smplh/tmp/neutral/model.npz --mano-left-fn deps/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn deps/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder deps/smplh/

echo
echo "Done!"
echo
echo "Deleting tmp files"
rm -rf deps/smplh/tmp/
echo 
echo "Done!"
