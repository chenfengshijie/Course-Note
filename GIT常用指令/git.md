## Git常用命令

- 设置Git名称和邮箱
```javascript
git config --global user.name "name"
git config --global user.email "email"
```
- 初始化Git仓库或者克隆仓库。克隆长裤会在当前文件夹下创建新的文件夹，里面包含.git,而git init会在当前文件夹下面创建.git仓库。

```javascript
git init
git clone https://github.com/some_respository/
```

- 添加文件进入仓库,可以在仓库的根目录下添加.gitignored文件，里面标注忽略管理的文件或者文件夹。

``` c
git add . # add all files to the repository
git add directory # add directory to repository
git add file # add file to repository
```
- 查看更改前后的区别,尽量每一次提交前都运行git diff HEAD 命令查看差别。

```c
git diff # all the files difference
git diff directory # the files in this directory
git diff HEAD # the difference in the working tree
```
- 创建分支和切换分支
```c
git checkout -b new_branch
git switch -c new_branch # latter is recommanded

git checkout new_branch
git switch new_branch # switch to new branch
```

- 合并分支，合并分支由-ff,-no-ff,-ff-only三个操作，ff是快速合并，它只是将节点指向两个公共祖先，并没有创建新节点，而-no-ff需要创建新节点，默认-ff。
- 可以通过git log --graph 以图表形式查看提交状态。

``` c
git merge -no-ff another_branch
```


- 回退操作，使用git reflog 可以查看所有的提交，然后根据这些hash值，使用reset命令回退，--hard,--soft,--mixed(默认),--hard会清空暂存区和工作目录下的所有内容，--soft会保留，--mix会保留工作目录，而清空暂存区。

``` c
git reset hash_of_log_commit
```
- 首先可以使用git remote add origin `https://github.com/`添加一个名为origin的远程仓库

- git push 和git pull 从远程仓库拉去和推送相应的内容
- 有的时候，会出现`fatal: refusing to merge unrelated histories`的提示，两个完全不相关的节点，Git不知道如何合并，可以在后面加上--allow-unrelated-histories参数，强制合并。
