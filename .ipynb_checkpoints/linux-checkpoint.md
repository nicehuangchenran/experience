# linux常用命令

## 基础

`echo`：打印内容

`pwd` 打印当前目录， **print working directory**

---------------------------------

`rm -rf ./*` 删除当前目录下所有文件和文件夹

`rm`：删除命令

`-r`：递归删除目录

`-f`：强制删除（不提示）

`./*`：当前目录下所有“非隐藏文件”

`rm -rf ./* ./.*` 将隐藏目录一起删除

----------------------------------------------------------

`export`：导出环境变量

把变量“传给当前 shell 启动的子进程”（例如 python、java、其他脚本）

```bash
export b=456
python -c 'import os; print(os.getenv("b"))'   # 456
```

```
export CUDA_VISIBLE_DEVICES=1
```

你的脚本里就这么用来指定 GPU，以及设置 `PYTHONPATH` 给 Python 进程读取。



`~/.zshrc`永久记住将变量写入这里，上述变量只在当前终端，关闭后没有，后续打开bash或者zsh会自动source

```bash
echo 'export MY_VAR="hello"' >> ~/.bashrc
source ~/.bashrc
```

## 查看显卡

`nvidia-smi`

```bash
chuangby@user-AS-4124GS-TNR:~$ ps -o user,pid,etime,cmd -p 3857870
USER         PID     ELAPSED CMD
fuxiang+ 3857870    01:26:42 python main_pretrainmammovlp.py --lambda_contrastive 1.0 --lambda_view 0.1 --output_dir /mnt/data/hfx/mammovlp_lambda_contrastive1_lambda_view01
chuangby@user-AS-4124GS-TNR:~$ ps -o pid,ppid,cmd -p 3857870
   PID    PPID CMD
3857870 3365493 python main_pretrainmammovlp.py --lambda_contrastive 1.0 --lambda_view 0.01 --out 
chuangby@user-AS-4124GS-TNR:~$ ps -o pid,ppid,cmd -p 3365493
    PID    PPID CMD
3365493 3363506 bash para.sh  # 说明这个命令是bash脚本在自动执行
$ last -n 20 #显示最近登录过的20个人
tmux(3363505).%0 #断线不掉任务
```

process:进程

