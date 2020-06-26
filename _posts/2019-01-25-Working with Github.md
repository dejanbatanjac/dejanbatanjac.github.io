---
published: true
layout: post
title: Creating Github Pull Requests
---

## Create a Pull Request 

Let's create a fork of the [PyTorch project](https://github.com/pytorch/pytorch) first, but this procedure is the same for any other project. 

## 1. You just need to click a button **[Fork]**.

and you will be redirected to github.com/USERNAME/pytorch.

![IMG](/images/github1.png)

Once redirected:

## 2. Clone the repo where you were redirected
    git clone https://github.com/dejanbatanjac/pytorch.git pytorch-fork
    cd pytorch-fork


## 3. Setup the fork to track upstream
Click the [Clone or download] button 

![IMG](/images/github3.png)

You will have HTTPS or SSH options:

    git remote add upstream https://github.com/pytorch/pytorch.git
    # or ...
    git remote add upstream git@github.com:USERNAME/pytorch.git
    
Now if you issue a command `git config -l` you should have `remote.upstream.url` set.
You can check the same with `git remote -v` that will return at this point:

    pytorch-fork>git remote -v
    origin  https://github.com/USERNAME/pytorch.git (fetch)
    origin  https://github.com/USERNAME/pytorch.git (push)
    upstream        https://github.com/pytorch/pytorch.git (fetch)
    upstream        https://github.com/pytorch/pytorch.git (push)


After day or two you will notice, your fork will be behind the master.


![IMG](/images/github2.png)

So what to do?

## 4. Make it even 

    cd pytorch-fork
    git checkout master
    git fetch upstream
    git checkout master
    git merge --no-edit upstream/master
    git push

Then you will get this message:

>This branch is even with pytorch:master. 

## 5. Create a branch and add your updates

    git checkout -b new-branch
    git push --set-upstream origin new-branch

## 6. Push your updates to origin

    git commit
    git push

## 7. Creating a *Pull request*

![IMG](/images/github4.png)

