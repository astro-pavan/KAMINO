cd ../external

mkdir phreeqc
cd phreeqc
INSTALLDIR=$PWD
cd ..

# unpacks and installs PHREEQC
tar -xvzf phreeqc-3.8.6-17100.tar.gz
cd phreeqc-3.8.6-17100
mkdir Release
cd Release
../configure --prefix=$INSTALLDIR
make
make install