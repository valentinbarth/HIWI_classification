#!/bin/bash
curl -sSL https://get.docker.com/rootless | sh
export PATH=/home/vbarth/bin:$PATH
export PATH=$PATH:/sbin
export DOCKER_HOST=unix:///run/user/1008/docker.sock
systemctl --user start docker 
# execute this file with ". config_rootless_docker.sh" without the ". " in the beginning it deos not work!!
