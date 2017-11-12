#!/usr/bin/env bash

# download from Google Drive (reference: https://stackoverflow.com/a/38937732/7322931)
ggID='1jFsh4Jh3s41B-_hPHe_VS9apkMmIWiNy'
ggURL='https://drive.google.com/uc?export=download'
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${filename}"

# extract
unzip bm_models.zip
rm -rf bm_models.zip
