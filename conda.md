# conda 学习笔记

## 环境

 `deactivate 环境名` 退出环境

`conda info --envs` 查看现有环境

`conda env list` 查看所有环境

`conda create --name mydrl python=3.7.0`创建环境
`conda remove --name your_environment_name --all`删除环境

`conda remove -n 环境名 --all` 删除环境

python包不可以用pip安装，只能用conda安装

## 安装依赖

```text
pip install 包名称 -i https://pypi.tuna.tsinghua.edu.cn/simple（清华镜像）
pip install 包名称 -i  https://pypi.doubanio.com/simple/ （豆瓣镜像）

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes  在安装时显示从哪个源安装
```

```
pip show numpy  查看当前安装的版本
pip install <package_name>==  查看所有可用版本
pip uninstall <package_name>
pip install <package_name>==<version>  卸载现有版本后安装新版本
```





```text
conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch 增加安装频道
conda config --show 查看频道
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 删除频道
```









`conda list` `pip list` 查看安装的包

`pip3 list`查看现在安装的包，在多个文件路径下都可以用

`py -3.10 -m pip list`电脑上有多个版本的pyhton，显示指定版本python的依赖包

`py -3.8 -m pip install `指定pyhton版本安装依赖包

`py -3.10 main.py`指定python版本运行程序

在snake-ai项目中gym=0.21.0安装失败，使用`pip install setuptools==65.5.0 "wheel<0.40.0"`

`conda clean --all --dry-run`查看那些包被删除，`conda clean --all`清理conda环境————系统级别，清理所有环



# cuda安装

`NVIDIA-SMI` ` nvcc -V` 查看基本信息

下载地址：https://download.pytorch.org/whl/cu +cuda版本号，官方版本对应关系https://pytorch.org/get-started/previous-versions/