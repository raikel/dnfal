#!/usr/bin/env bash
WEIGHTS_FACE_DETECTOR=weights_face_detector.pth
WEIGHTS_FACE_ENCODER=weights_face_encoder.pth
WEIGHTS_FACE_MARKER=weights_face_marker.npy
WEIGHTS_GENDERAGE_PREDICTOR=weights_genderage_predictor.pth
mkdir -p models
cd models/

if [[ ! -f "$WEIGHTS_FACE_DETECTOR" ]]; then
    wget -O ${WEIGHTS_FACE_DETECTOR} "https://www.dropbox.com/s/z4l7rasz0ydjt9f/weights_face_detector.pth?dl=0"
fi

if [[ ! -f "$WEIGHTS_FACE_ENCODER" ]]; then
    wget -O ${WEIGHTS_FACE_ENCODER} "https://www.dropbox.com/s/6a3eztge2ghci3q/weights_face_encoder.pth?dl=0"
fi

if [[ ! -f "$WEIGHTS_FACE_MARKER" ]]; then
    wget -O ${WEIGHTS_FACE_MARKER} "https://www.dropbox.com/s/2ne3ronmdqgxsca/weights_face_marker.npy?dl=0"
fi

if [[ ! -f "$WEIGHTS_GENDERAGE_PREDICTOR" ]]; then
    wget -O ${WEIGHTS_GENDERAGE_PREDICTOR} "https://www.dropbox.com/s/v4nppzp6xdb4jpg/weights_genderage_predictor.pth?dl=0"
fi