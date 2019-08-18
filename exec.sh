#!/bin/bash

nvidia-docker run --rm -v $GITHUB/tanimutomo/DeepTextures:/workspace -it texsyn $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
