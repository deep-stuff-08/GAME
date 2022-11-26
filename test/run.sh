export LD_LIBRARY_PATH="../lib"
g++ test.cpp -I "../src/include" -o run -lGAME
./run
rm run