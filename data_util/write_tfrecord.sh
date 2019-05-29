#!/bin/sh
python write_tfrecord.py 0 1000000 &
python write_tfrecord.py 1000000 1800000 &
