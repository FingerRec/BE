#!/usr/bin/env bash
# !/bin/bash

src_path=`readlink -f $1`
dst_path=`readlink -f $2`

rar_files=`find $src_path -name '*.rar'`
IFS=$'\n'; array=$rar_files; unset IFS
for rar_file in $array; do
    file_path=`echo $rar_file | sed -e "s;$src_path;$dst_path;"`
    ext_path=${file_path%/*}
    if [ ! -d $ext_path ]; then
        mkdir -p $ext_path
    fi
    unrar x $rar_file $ext_path
done

# bash utils/data_process/unrar_hmdb51_sta.sh /data1/DataSet/hmdb51_sta /data1/DataSet/hmdb51_sta_new