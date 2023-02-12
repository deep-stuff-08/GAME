cls

del *.exe

del *.obj

del Log.txt

cl.exe /c /DEBUG:FULL /Iheader /ID:\Astromedicomp\RTR\01_OpenGL\02_PP\01_Windows\include /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include" src/*.cpp 

rc.exe /i header resource\OGL.rc

link.exe *.obj resource\*.res /SUBSYSTEM:WINDOWS /LIBPATH:D:\Astromedicomp\RTR\01_OpenGL\02_PP\01_Windows\lib\Release\x64 /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" OpenCL.lib user32.lib gdi32.lib kernel32.lib
