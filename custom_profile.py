"""
测试.py文件各个函数的运行时间
"""
import pstats

# 在cmd中使用python -m cProfile -o result.cprofile  main.py

# 创建 Stats 对象
p = pstats.Stats('./result.cprofile')
# 按照运行时间和函数名进行排序
# 按照函数名排序，只打印前n行函数(其中n为print_stats(n)的输入参数)的信息,
p.strip_dirs().sort_stats("cumulative", "name").print_stats(15)
# 参数还可为小数, 表示前n(其中n为一个小于1的百分数, 是print_stats(n)的输入参数)的函数信息
# p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)
# 查看调用main()的函数
# p.print_callers(0.5, "main")
# 查看main()函数中调用的函数
# p.print_callees("main")

# pip安装snakeviz后，在cmd里运行如下命令：
# snakeviz result.out
