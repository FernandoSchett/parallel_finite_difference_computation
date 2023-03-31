rm -rf bin include lib 
export CWPROOT=$PWD
cd src 
make install 
make xtinstall 
cd -
cp ../util/* include/
