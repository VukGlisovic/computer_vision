#!/bin/bash

# Load the environment variables
source .env

# Create the machine
gcloud compute instances create "hdr-image-creation" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="e2-custom-8-16384" \
    --image-family="ubuntu-2404-lts-amd64" \
    --image-project="ubuntu-os-cloud" \
    --boot-disk-size="20GB" \
    --boot-disk-type="pd-standard" \
    --network="default" \
    --provisioning-model=SPOT \
    --max-run-duration=1h \
    --instance-termination-action=STOP \
    --scopes=default \
    --async
