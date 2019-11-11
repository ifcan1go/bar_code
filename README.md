# bar_code
require

operncv

pip install opencv-python

run

运行main.py对图片进行处理
共需要7个参数
1. -low 用于识别图片条带，低阈值
2. -high 用于识别图片条带，高阈值
3. -img_name 需要处理的文件名（可以使用绝对路径+文件名）
4. -img_path 处理处理的文件路径（如果使用绝对路径，此项为空）
5. -output_path 输出文件路径（输出的图片名称与需处理的文件名相同）
6. -mark_in_name 内标图片名（可以使用绝对路径+文件名，与可以是img_path+makrk_in_name)
7. -makr_out_name 外表图片名（可以使用绝对路径+文件名，与可以是img_path+makrk_in_name)
