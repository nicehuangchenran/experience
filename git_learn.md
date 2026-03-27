# git

## 正在用

```
git checkout -- omin代码说明.md
```

本质作用是：

> **把这个文件恢复成“最后一次 commit 时的状态”，并丢弃你所有本地修改**

## 基本

```
工作区 → 暂存区 → 本地仓库 → 远程仓库
```

对应命令：

| 操作             | 命令         |
| ---------------- | ------------ |
| 添加文件到暂存区 | `git add`    |
| 提交到本地仓库   | `git commit` |
| 推送到远程仓库   | `git push`   |

```powershell
git init # 当前文件夹下初始git
# 关联远程仓库
git remote add origin https://github.com/username/repo.git  # origin是远程仓库的关联名称
git add .
git commit -m "comments"
```

Pull Request：“把一个分支合并到另一个分支”的请求

`git pull`的时候 会自动合并

## 分支操作

### 查看

```
git branch
```

👉 查看本地分支（当前分支前有 `*`）

```
git branch -a
```

👉 查看**所有分支（本地 + 远程）**

```
git branch -r
```

👉 只看远程分支

### 创建分支

```
git branch dev
```

👉 创建分支（但不会切换）

### 切换分支

```
git checkout dev
```

### 删除分支

```
git branch -d dev
```

👉 删除本地分支（安全删除，未合并会报错）

------

```
git branch -D dev
```

👉 强制删除（不管是否合并）

### 重命名分支

```
git branch -m old_name new_name
```

### 合并分支（merge）

```
git merge dev
```

👉 把 `dev` 合并到当前分支

### 远程分支相关

`git remote -v` 查看关联的远程仓库

推送分支到远程

```
git push origin dev
```

------

推送并建立跟踪关系（推荐）

```
git push -u origin dev
```

👉 以后可以直接 `git push`

------

删除远程分支

```
git push origin --delete dev
```

------

拉取远程分支

```
git pull origin dev
```

git branch -r 

git branch 查看本地repo的branch

git branch -d local_branch_name 删除本地branch

`git branch -m 新分支名称`本地branch改名

git checkout branch-name 切换branch

git checkout -b first-branch 增加一个branch

git branch -a 查看本地和远程repo的branch

`git push remote_name -d remote_branch_name` 删除远程repo的branch

## 本地和远程仓库

```
git clone -b 分支名 仓库地址
```

例如：

```
git clone -b dev https://github.com/username/repo.git
```

👉 效果：

- 克隆整个仓库
- 并**自动切换到 `dev` 分支**

------

二、只克隆指定分支（更高效）

```
git clone -b dev --single-branch 仓库地址
```

👉 区别：

- `-b dev`：指定分支
- `--single-branch`：**只拉这个分支的历史（更快、更省空间）**

## 查看commit记录

`git log origin/main`

`git log`查看本地commit记录

`git reset --soft ef0d6eda9bacb6c114d0bf2f0bc4aef7b8087cc4`恢复到某一个commit

## 提高clone速度

`git clone https://gitclone.com/github.com/znxlwm/pytorch-generative-model-collections.git`

一时不知道怎么提高pull 速度

设置代理

git config --global http.proxy http://192.168.230.1:7890
git config --global https.proxy https://192.168.230.1:7890

git config --global http.proxy http://192.168.183.1:7890
git config --global https.proxy https://192.168.183.1:7890

git config --global --get http.proxy 查看设置的代理

重置取消代理

git config --global --unset http.proxy
git config --global --unset https.proxy

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



