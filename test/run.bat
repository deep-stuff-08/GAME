
cl.exe /c /EHsc /I..\src\include /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include" test.cpp

link.exe test.obj /LIBPATH:..\lib\ GAME.lib