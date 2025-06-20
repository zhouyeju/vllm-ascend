#!/bin/bash
HOST=$1
PORT=$2
PREFILL_PORT=$3
DECODE_PORT=$4

python ./simple_pd_proxy_server.py \
    --host ${HOST} \
    --port ${PORT} \
    --prefiller-host ${HOST} \
    --prefiller-port ${PREFILL_PORT} \
    --decoder-host ${HOST} \
    --decoder-port ${DECODE_PORT} &