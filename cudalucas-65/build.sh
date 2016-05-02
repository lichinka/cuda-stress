#!/bin/sh

# 
# compilation script 
#
case $( hostname ) in 
    *daint* | *santis*)
        MODS="PrgEnv-gnu cudatoolkit"
        MAKE_INC="make.daint"
        ;;
    *greina*)
        MODS="cuda70/toolkit/7.0.28"
        MAKE_INC="make.greina"
        ;;
    *)
        echo "Don't know how to compile here. Exiting."
        exit 1
        ;;
esac

echo -n "Checking modules on $( hostname ) ... "
for m in ${MODS}; do
    if [ -z "$( echo ${LOADEDMODULES} | grep ${m} )" ]; then
        echo -e "${m} is missing"
        exit 1
    fi
done
echo "ok"

echo "Building on $( hostname ) ..."
cat ${MAKE_INC} Makefile > /tmp/.makefile
make -Bf /tmp/.makefile
