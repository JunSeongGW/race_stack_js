#!/bin/bash

if [ $# -ne 1 ]
then
        echo "usage: $0 <repo:tag>"
        exit
fi

IMAGE=$1
USER_ID=`/usr/bin/id -u $USER`

XSOCK=/tmp/.X11-unix
XAUTH=$XAUTHORITY 
VULKAN=/usr/share/vulkan
SHARED_DOCK_DIR=/home/misys/shared_dir
SHARED_HOST_DIR=/home/$USER/shared_dir

DEVICES="--device /dev/snd --device /dev/dri"

VOLUMES="--volume=$XSOCK:$XSOCK:ro	 
	 --volume=$XAUTH:$XAUTH:ro
	 --volume=$SHARED_HOST_DIR:$SHARED_DOCK_DIR:rw
	 --volume=/media:/media:rw
	 --volume=/dev/bus/usb:/dev/bus/usb:rw
	 --volume=/dev/input:/dev/input:rw
	 --volume=/dev/shm:/dev/shm:rw
	 --volume=/dev/ttyUSB0:/dev/ttyUSB0:ro
	 --volume=/dev/ttyACM0:/dev/ttyACM0:rw
	 --volume=/run/udev:/run/udev:rw
	 --volume=$VULKAN:$VULKAN:rw"
	 #--volume=$VULKAN:$VULKAN:rw
	 
ENVIRONS="--env DISPLAY=$DISPLAY
	  --env SDL_VIDEODRIVER=x11
	  --env XAUTHORITY=$XAUTHORITY"

GPU='--gpus all,"capabilities=graphics,utility,display,video,compute"' 
#GPU='--gpus all'


FULL_CMD="docker run -it \
	-u $USER_ID \
	--privileged \
	--net=host \
	$GPU \
	$DEVICES \
	$ENVIRONS \
	$VOLUMES \
	$IMAGE"	

if [ $USER_ID -ne 1000 ]; then
	echo "host uid must be 1000; switch user or edit command in script"
	exit 1
fi

if [ ! -d $SHARED_HOST_DIR ]; then
	mkdir $SHARED_HOST_DIR
fi

echo $FULL_CMD

$FULL_CMD
