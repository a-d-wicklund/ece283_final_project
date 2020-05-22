# Some tips n stuff

## Git  
### Clone the repo into your local directory and create a new branch
Click on the green button on Github that says "Clone or Download". Copy the link under the header "Clone with HTTPS" (unless you have ssh set up already). Then go to a directory on your computer where you want the main project folder to go. Assuming you have git installed on your computer, open a terminal and enter  

`git clone <url-that-you-just-copied>`  

Now the project directories, along with all the files have been loaded into the folder. You now want to create your own branch. In the project directory:  

`git checkout -b <name-of-your-branch>`  

This will create a new branch off of the master and put you on it. Now, as long as you're on this branch, any changes you make to files will only be reflected locally and will not effect the master branch. Make sure that you're actually on the new branch and then sync up your local branch with a new branch with the same name on Github with:  

`git branch`  
`git push --set-upstream origin <local-branch-name>`  

Git push just "pushes" whatever changes you've made locally to the remote (Github) repo. Now, whenever you're on this branch on your computer and you `git push` changes or `git pull`, it does so to and from the branch with the same name on Github.  
To switch back to the master branch:  

`git checkout master`  

## Python 2to3

The project we're working on was originally written for Python 2, but we want to run ours with Python 3. Some of the conversion can't be caught super quickly, like integer division, but some of it can. We use the command `2to3` for the latter case. Go to your terminal and navigate to the directory where you have .py files that need to be converted. Run  

`2to3 <file-name>`  
to see which changes need to be made for conversion and  
`2to3 -w <file-name>`  
to actually make those changes and write them to the file.