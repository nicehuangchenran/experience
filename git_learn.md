# git

## 分支操作

git branch -r 

git branch 查看本地repo的branch

git branch -d local_branch_name 删除本地branch

`git branch -m 新分支名称`本地branch改名

git checkout branch-name 切换branch

git checkout -b first-branch 增加一个branch

git branch -a 查看本地和远程repo的branch

`git push remote_name -d remote_branch_name` 删除远程repo的branch

## 本地和远程仓库

`git log`查看本地commit记录

`git reset --soft ef0d6eda9bacb6c114d0bf2f0bc4aef7b8087cc4`恢复到某一个commit

## 查看commit记录

`git log origin/main`

## 本地修改后pull

编辑器是VIM	

输入注释

- 按下 `ESC` 键退出编辑模式。
- 输入 `:wq`（意为 write and quit，即保存并退出）。
- 按 `Enter` 键提交这个命令。

这会保存你的注释并关闭编辑器，完成提交。

## 提高clone速度

`git clone https://gitclone.com/github.com/znxlwm/pytorch-generative-model-collections.git`

一时不知道怎么提高pull 速度

## 其他

git init 在此目录下创建一个git，生成.git文件

git remote add origin git@github.com:RuiTan/test.git 添加远程仓库，origin是远程仓库的别名

git remote -v 查看本地仓库所链接的远程仓库，

	git remote -v
	origin  git@github.com:nicehuangchenran/Test.git (fetch)
	origin  git@github.com:nicehuangchenran/Test.git (push)

`git remote remove origin`删除与远程仓库的链接而不影响远程仓库

git pull origin main 从origin远程仓库拉取main branch到本地

git add . 将分支中的内容上传到本地缓冲区（index）

git status 查看当前本地repo分支状态

git commit -m "增加了一个git总结文档" commit到本地repo

git push -u origin first-branch 发起push，项目创建者可以据此创建pull request,创建pull request后可以选择merge到base branch

`git push origin -u  hcr --verbose`可以输出更多信息，在没有反应的时候使用

`git config --global user.name "你的用户名"``git config --global user.email "你的邮箱"`设置用户名和邮箱

设置代理

git config --global http.proxy http://192.168.230.1:7890
git config --global https.proxy https://192.168.230.1:7890

git config --global http.proxy http://192.168.183.1:7890
git config --global https.proxy https://192.168.183.1:7890

git config --global --get http.proxy 查看设置的代理

重置取消代理

git config --global --unset http.proxy
git config --global --unset https.proxy
