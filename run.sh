python inference.py --device cpu --precision float32
python inference.py --device cuda --precision float16 --jit --nv_fuser --channels_last 1 --profile
python inference.py --device xpu --precision float16 --jit --channels_last 1
