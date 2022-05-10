# Download rendered PhotoShapes chairs dataset.
DATA_ROOT=/home/nas2_userF/sungwonhwang/ws/data
PROJ_ROOT=/home/nas2_userF/sungwonhwang/ws/LocalNeRF/model/cnerf

#mkdir -p ${DATA_ROOT}/photoshapes
#cd ${DATA_ROOT}/photoshapes

#wget editnerf.csail.mit.edu/photoshapes.zip
#echo 'Unzipping photoshapes dataset'
#unzip -q photoshapes.zip
#mv photoshapes/* .

#cd ../../

# Download and process rendered CARLA cars dataset.
#mkdir -p ${DATA_ROOT}/carla/carla_images
#cd ${DATA_ROOT}/carla/carla_images

#wget https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla.zip
#echo 'Unzipping CARLA dataset'
#unzip -q carla.zip

#cd ..
#wget https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla_poses.zip
#unzip -q carla_poses.zip

#cd ../../
#echo 'Formatting CARLA dataset'
python ${PROJ_ROOT}/utils/setup_cars.py ${DATA_ROOT}/carla

# Download and process rendered Dosovitskiy chairs dataset.
#mkdir -p ${DATA_ROOT}/dosovitskiy_chairs/
#cd ${DATA_ROOT}/dosovitskiy_chairs

#wget https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar
#echo 'Unzipping Dosovitskiy chairs dataset'
#tar -xf rendered_chairs.tar
#cd ../../

echo 'Formatting Dosovitskiy chairs dataset'
python ${PROJ_ROOT}/utils/setup_dosovitskiy.py ${DATA_ROOT}/dosovitskiy_chairs
