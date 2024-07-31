rm -rf /home/oleg/kodII/mrcv/build/*
rm -rf /home/oleg/kodII/mrcv/examples/libfirst/build/*

cmake -B/home/oleg/kodII/mrcv/build -H/home/oleg/kodII/mrcv
make -C/home/oleg/kodII/mrcv/build
sudo make -C/home/oleg/kodII/mrcv/build install
sudo ldconfig

cmake -B/home/oleg/kodII/mrcv/examples/libfirst/build -H/home/oleg/kodII/mrcv/examples/libfirst
make -C/home/oleg/kodII/mrcv/examples/libfirst/build
