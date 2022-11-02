# notice for input:
```
the dataloader read file for input, the param<first_n_byte> default value is 2000000, it means read 2million byte once time, but the file only 22k,so you can adjust this param to a more suitable value.
```



python inference.py --device cpu --precision float32
python inference.py --device cuda --precision float16 --jit --nv_fuser --channels_last 1 --profile
python inference.py --device xpu --precision float16 --jit --channels_last 1
