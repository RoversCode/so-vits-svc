name="sunyanzi"
python preprocess/rename_and_slice.py  \
 --audio_dir ckpts/$name/raw_audio \
 --sr 44100 \
 --db_thresh -40 \
 --sliced_min_length 5000 \
 --min_interval 300 \
 --audio_min_length 2000 \
 --hop_size 10 \
 --max_sil_kept 500
