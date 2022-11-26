rm lib/libGAME.so
g++ src/game.cpp src/clcontext.cpp src/clhelper/clhelper.cpp -lOpenCL --shared -fPIC -o lib/libGAME.so