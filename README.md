# range_only_robotics


## Ipopt

To get a clean working setup, follow the tutorial:  https://coin-or.github.io/Ipopt/INSTALL.html#COINBREW

Note: As of now I have no way of accessing access to any of the HSL linear solvers and need to solve everything using MUMPS

mkdir opt
cd opt
touch coinbrew

' Edit coinbrew and copy and paste in the code from coin-or repository '

bash coinbrew fetch Ipopt
bash coinbrew build Ipopt --enable-debug

This will get you a working setup but you are limited to MUMPS!

SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
app->Options()->SetStringValue("linear_solver", "mumps");


Note: I had issue installing Ipopt with HSL. Still work in progress. This is the most recent attempt

1. Go to http://hsl.rl.ac.uk/ipopt.
2. Choose whether to download either the Archive code or the HSL Full code. To download, select the relevant "source" link.
3. Follow the instructions on the website, read the license, and submit the registration form.
4. After downloading the "Archive Code" I only had access to a precompiled HSL library. I called the folowing file as recommended by the Ipopt tutorial

./configure --prefix=/home/alan/opt/Ipopt/3.10.0 --with-blas --with-blas-incdir=/usr/include --with-blas-lib=/usr/lib/libblas --with-hsl-incdir=/home/alan/opt/coinhsl/include --with-hsl-lib=/home/alan/opt/coinhsl/lib --with-lapack --with-lapack-incdir=/usr/include --with-lapack-lib=/usr/lib/lapack 