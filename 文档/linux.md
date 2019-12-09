##  ssh到linux 时后台执行脚本的两种方式
###  背景知识
在linux中，ssh到linux的会话窗口对应一个父进程，在这个ssh会话中执行的所有命令都是这个父进程的子进程，当ssh会话断开后，父进程终止，父进程的子进程也会终止，所以大体来说有两种解决办法：1. 加强父进程，如将ssh会话设置为不会自动断开，2. 将子进程赋给其他父进程，比如将子进程赋给一个pid为1的父进程

###  nohup 命令
这个命令是linux（acl）自带的，相当于将程序在后台执行，并且将terminal输出重定向，默认重定向到当前文件夹下的nohup.txt.

形式为

    nohup python -u test.py >> ~/output.txt &

其中 -u 是python自带的参数，表示unbufferd
通常python的输出是有一个buffer，就是说print不会马上输出，用这个命令可以让print，马上输出，方便查看。
& 符号表示命令部分的终止。

nohup的缺点就是没有tty交互，tmux和screen规避了这个缺点。

###  screen

可能需要自行安装。screen是一个虚拟多终端工具。

两种方法启动
1. 先screen进入一个新的screen窗口，然后执行命令。
2. 直接screen python test.py
3. 
在每个screen session 下，所有命令都以 ctrl+a(C-a) 开始（这些命令可以判断是否在screen session里面）

隐藏当前screen窗口

    ctrl + a, ctrl + z
    fg命令回到
查看有哪些 screen 在运行

    screen -ls

进入某个screen

    screen -r 13054
    进入某个screen后，ctrl+c关闭screen

要关闭某个 screen，可以用如下：

    kill ps_ID