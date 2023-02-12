// Header Files
#include<windows.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<sstream>
#include<memory>
#include"OGL.h"

// OpenCL Header Files
        //glVertexAttribPointer(ATTRIBUTE_POSITION,4,GL_FLOAT,GL_FALSE,0,NULL);
        //glEnableVertexAttribArray(ATTRIBUTE_POSITION);
#include<CL/cl.h>
#include<CL/cl_gl_ext.h>

// OpenGL Header Files
#include<GL/glew.h> // this must be above gl.h
#include<GL/gl.h>

#include"vmath.h"
using namespace vmath;

#define WIN_WIDTH 640
#define WIN_HEIGHT 360

// OpenGL Libraries

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"OpenGL32.lib")
#pragma comment(lib,"glu32.lib")

// Global Variable

unsigned int mesh_width = 256;
unsigned int mesh_height = 256;

HWND ghwnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
BOOL gbFullScreen = FALSE;
BOOL gbActiveWindow = FALSE;
FILE *gpFile = NULL;
TCHAR str[125];

// Programable Pipeline related variables
GLuint shader_program_object;

enum
{
    ATTRIBUTE_POSITION = 0,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEXTURE0,
};

// FPS Counter Logic
int fpsCount = 0;
int fpsLimit = 1;
int g_index = 0;
float avgFps = 0.0f;
unsigned int frameCount = 0;


// OpenCL Vars

cl_platform_id cpPlatform;
cl_context cxGPUcontext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;
cl_kernel ckKernel;
cl_mem vbo_cl;
cl_program cpProgram;
cl_int ciErrNum;
const char* cSrcCL = NULL;
size_t szGlobalWorkSize[] = {mesh_width,mesh_height};

// OpenGL Vars
GLuint vao;
GLuint vbo_vertex,vbo_cpu;
GLint mvpUniformMatrix;
mat4 perspectiveProjectionMatrix;

BOOL onGPU = FALSE;

GLfloat angle_triangle = 0.0f;

GLfloat* buffer;

// Global Function Declarations
LRESULT CALLBACK WndProc(HWND,UINT,WPARAM,LPARAM);
void runCPU();
void runKernel();
void CreateVBO();
void ComputeFPS();

// Main Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,LPSTR lpszCmdLine,int iCmdShow)
{
    // Function Declaration
    int initialize(void);
    void display(void);
    void update(void);
    void uninitialize(void);
    // Variable Declaration
    WNDCLASSEX wndclass;
    HWND hwnd;
    MSG msg;
    TCHAR szAppName[] = TEXT("MyWindow");
    BOOL bDone = FALSE;
    int iRetVal = 0;

    // Code

    MessageBox(NULL, TEXT("Creation Of Log File Failed.\nExitting ..."), TEXT("File I/O Error"), MB_ICONERROR);

    if(fopen_s(&gpFile,"Log.txt","w") != 0)
    {
        MessageBox(NULL,TEXT("Creation Of Log File Failed.\nExitting ..."),TEXT("File I/O Error"),MB_ICONERROR);
        exit(0);
    }
    else
    {
        //fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile,"Creating Log File.\n");
        fclose(gpFile);
    }

    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    int x = (screenWidth -  WIN_WIDTH) / 2;
    int y = (screenHeight - WIN_HEIGHT) / 2;

    // Initialization Of WNDCLASSEX Structure
    wndclass.cbSize         = sizeof(WNDCLASSEX);
    wndclass.style          = CS_HREDRAW|CS_VREDRAW|CS_OWNDC;
    wndclass.cbClsExtra     = 0;
    wndclass.cbWndExtra     = 0;
    wndclass.lpfnWndProc    = WndProc;
    wndclass.hInstance      = hInstance;
    wndclass.hbrBackground  = (HBRUSH)GetStockObject(BLACK_BRUSH); 
    wndclass.hIcon          = LoadIcon(hInstance,MAKEINTRESOURCE(BAT_ICON));
    wndclass.hCursor        = LoadCursor(NULL,IDC_ARROW);
    wndclass.lpszClassName  = szAppName;
    wndclass.lpszMenuName   = NULL;
    wndclass.hIconSm        = LoadIcon(hInstance,MAKEINTRESOURCE(BAT_ICON));

    // Registering Above WndClass
    RegisterClassEx(&wndclass);

    // Create Window
    hwnd = CreateWindowEx(  
                            WS_EX_APPWINDOW,
                            szAppName,
                            TEXT("Sine Wave"),
                            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
                            x,
                            y,
                            WIN_WIDTH,
                            WIN_HEIGHT,
                            NULL,
                            NULL,
                            hInstance,
                            NULL
                        );

    ghwnd = hwnd;

    iRetVal = initialize();

    if(iRetVal == -1)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile,"choose pixel format failed");
        fclose(gpFile);
        uninitialize();
    }

    if(iRetVal == -2)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile,"set pixel format failed");
        fclose(gpFile);
        uninitialize();        
    }

    if(iRetVal == -3)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile,"Create wgl context failed");
        fclose(gpFile);
        uninitialize();        
    }

    if(iRetVal == -4)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile,"make current context failed");
        fclose(gpFile);
        uninitialize();        
    }

    if(iRetVal == -5)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile,"glewinit failed ");
        fclose(gpFile);
        uninitialize();        
    }

    if (iRetVal == -7)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "OpenCL Error ");
        fclose(gpFile);
        uninitialize();
    }

    // Show window
    ShowWindow(hwnd,iCmdShow);

    // Foregrounding and Focusing The Window
    // ghwnd or hwnd will work but hwnd is for local functions.
    SetForegroundWindow(hwnd);

    SetFocus(hwnd);

    // Special loop
    while(!bDone)
    {
        if(PeekMessage(&msg,NULL,0,0,PM_REMOVE))
        {
            if(msg.message == WM_QUIT)
                bDone = TRUE;
            else
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
        else
        {
            if(gbActiveWindow)
            {
                // Render The Scene
                display();
                // Update the Scene
                update();
            }
        }
    }
    uninitialize();
    return (int)msg.wParam;
}

// Callback Function
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    // Local Function Declaration
    void ToogleFullScreen(void);
    void resize(int,int);
    void uninitialize(void);
    // Local Variable 
    // Code
    switch(iMsg)
    {
        case WM_CREATE:
            fopen_s(&gpFile, "Log.txt", "a");
            fprintf(gpFile,"In WM_CREATE Message.\n");
            fclose(gpFile);
            sprintf(str,"!!! Press F To Enter FullScreen !!!");
        break;

        case WM_CHAR:
            switch(wParam)
            {
                case 'F':
                    case 'f':
                        fopen_s(&gpFile, "Log.txt", "a");
                        fprintf(gpFile,"In ToogleFullscreen.\n");
                        fclose(gpFile);
                        ToogleFullScreen();
                break;
               case 'I':
                    case'i':
                        mesh_width = mesh_width * 2;
                        mesh_height = mesh_height * 2;

                        if (onGPU)
                        {
                            szGlobalWorkSize[0] = mesh_width;
                            szGlobalWorkSize[1] = mesh_height;
                            //clReleaseMemObject(vbo_cl);
                            CreateVBO();
                        }
                        else
                            buffer = (GLfloat*)realloc(buffer, mesh_width * mesh_height * 4 * sizeof(GLfloat));                    
                        fopen_s(&gpFile, "Log.txt", "a");
                        fprintf(gpFile, "width %d height %d\n",mesh_width,mesh_height);
                        fclose(gpFile);

                 break;
                default:
                break;
            }
        break;

        case WM_SETFOCUS:
            fopen_s(&gpFile, "Log.txt", "a");
            fprintf(gpFile,"Set Focus True.\n");
            fclose(gpFile);
            gbActiveWindow = TRUE;
        break;

        case WM_KILLFOCUS:
            fopen_s(&gpFile, "Log.txt", "a");
            fprintf(gpFile,"Set Focus False.\n");
            fclose(gpFile);
            //gbActiveWindow = FALSE;
        break;

        case WM_ERASEBKGND:
            return 0;
        break;

        case WM_KEYDOWN:
            if(wParam == VK_ESCAPE)
            {
                fopen_s(&gpFile, "Log.txt", "a");
                fprintf(gpFile,"Sending WM_CLOSE.\n");
                fclose(gpFile);
                DestroyWindow(hwnd);
            }
            if (wParam == VK_SPACE)
            {
                onGPU = !onGPU;
                if (onGPU)
                {
                    szGlobalWorkSize[0] = mesh_width;
                    szGlobalWorkSize[1] = mesh_height;
                    //clReleaseMemObject(vbo_cl);
                    CreateVBO();
                }
            }
        break;

        case WM_SIZE:
            //fprintf(gpFile,"In WM SIZE message.\n");
            resize(LOWORD(lParam),HIWORD(lParam));
        break; 

        case WM_CLOSE:
            fopen_s(&gpFile, "Log.txt", "a");
            fprintf(gpFile,"In WM CLOSE message.\n");
            fclose(gpFile);
            DestroyWindow(hwnd);
        break;

        case WM_DESTROY:
            //uninitialize();
            PostQuitMessage(0);
        break;

        default:
            break;
    }
    return DefWindowProc(hwnd,iMsg,wParam,lParam);
}

void ToogleFullScreen(void)
{
    // Varriable Declarations
    static DWORD dwStyle;
    static WINDOWPLACEMENT wp;
    MONITORINFO mi;

    // Code
    wp.length = sizeof(WINDOWPLACEMENT);
    if(gbFullScreen == FALSE)
    {
        dwStyle = GetWindowLong(ghwnd,GWL_STYLE);
        if(dwStyle & WS_OVERLAPPEDWINDOW)
        {
            mi.cbSize = sizeof(MONITORINFO);
            if(GetWindowPlacement(ghwnd,&wp) && GetMonitorInfo(MonitorFromWindow(ghwnd,MONITORINFOF_PRIMARY),&mi))
            {
                SetWindowLong(ghwnd,GWL_STYLE,(dwStyle & (~WS_OVERLAPPEDWINDOW)));
                SetWindowPos(   ghwnd,HWND_TOPMOST,mi.rcMonitor.left,mi.rcMonitor.top,
                                mi.rcMonitor.right - mi.rcMonitor.left,
                                mi.rcMonitor.bottom - mi.rcMonitor.top,
                                SWP_NOZORDER|SWP_FRAMECHANGED);
            }

            ShowCursor(FALSE);
            gbFullScreen = TRUE;
        }
    }
    else
    {
        SetWindowLong(ghwnd,GWL_STYLE,dwStyle|WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(ghwnd,&wp);
        SetWindowPos(ghwnd,HWND_TOP,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOOWNERZORDER|SWP_NOZORDER|SWP_FRAMECHANGED);
        ShowCursor(TRUE);
        gbFullScreen = FALSE;
    }
}

int initialize(void)
{

    buffer = (float*)malloc(sizeof(float) * mesh_width * mesh_height * 4);

    // Function Declarations
    void resize(int, int);
    void printGLInfo(void);
    void uninitialize(void);
    // Variable Declarations
    PIXELFORMATDESCRIPTOR pfd;
    int iPixelFormatIndex = 0;

    // Code

    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cRedBits = 8;
    pfd.cGreenBits = 8;
    pfd.cBlueBits = 8;
    pfd.cAlphaBits = 8;
    pfd.cDepthBits = 32; // 24 is also allowed

    // Get DC
    ghdc = GetDC(ghwnd);

    // Choose Pixel Format
    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);

    if (iPixelFormatIndex == 0)
        return -1;

    // Set The Choosen Pixel Format
    if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
        return -2;

    // Create OpenGL Rendering Index

    ghrc = wglCreateContext(ghdc);

    if (ghrc == NULL)
        return -3;

    // Make Rendering as current context and rendering context
    if (wglMakeCurrent(ghdc, ghrc) == FALSE)
        return -4;

    // Here Starts OpenGL Code :

    //glew initialization
    if(glewInit() != GLEW_OK)
        return -5;

    // print opengl info
    //printGLInfo();

    // Vertex Shader
    const GLchar * vertex_shader_src = 
        "#version 460 core " \
        "\n" \
        "in vec4 a_position;\n" \
        "out vec4 out_color;\n" \
        "\n" \
        "uniform mat4 u_mvpMatrix;\n" \
        "\n" \
        "void main(void)\n" \
        "{\n"  \
        "   gl_Position = u_mvpMatrix * a_position;\n" \
        "}\n";

    // creating vertex shader object
    GLuint vertex_shader_object = glCreateShader(GL_VERTEX_SHADER);
    // link shader src to shader object
    glShaderSource(vertex_shader_object,1,(const GLchar**)&vertex_shader_src,NULL);
    // Compile Shader
    glCompileShader(vertex_shader_object);
    
    // Error checking
    GLint status = 0;
    GLint infoLogLength;
    char * log = NULL;

    glGetShaderiv(vertex_shader_object, GL_COMPILE_STATUS, &status);
    if(status == GL_FALSE)
    {
        glGetShaderiv(vertex_shader_object,GL_INFO_LOG_LENGTH,&infoLogLength);
        if(infoLogLength > 0)
        {
            log = (char*)malloc(infoLogLength);
            if(log != NULL)
            {
                GLsizei written;
                glGetShaderInfoLog(vertex_shader_object,infoLogLength,&written,log);
                fopen_s(&gpFile, "Log.txt", "a");
                fprintf(gpFile,"Vertex Shader Log : %s\n",log);
                fclose(gpFile);
                free(log);
                uninitialize();
            }
        }
    }

    // fragment shader
    const GLchar * fragment_shader_src = 
    "#version 460 core" \
    "\n" \
    "uniform vec4 out_color;\n" \
    "out vec4 FragColor;\n"\
    "\n"\
    "void main(void)\n" \
    "{\n"\
    "   FragColor = out_color;\n"\
    "}\n";

    GLuint fragment_shader_object = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader_object,1,(const GLchar**)&fragment_shader_src,NULL);
    glCompileShader(fragment_shader_object);

    // Shader Program Object
    shader_program_object = glCreateProgram();
    glAttachShader(shader_program_object,vertex_shader_object);
    glAttachShader(shader_program_object,fragment_shader_object);
    glBindAttribLocation(shader_program_object,ATTRIBUTE_POSITION,"a_position");
    glLinkProgram(shader_program_object);

    status = 0;
    infoLogLength = 0;
    log = NULL;

    glGetProgramiv(shader_program_object,GL_LINK_STATUS,&status);

    if(status == GL_FALSE)
    {
        glGetProgramiv(shader_program_object,GL_INFO_LOG_LENGTH,&infoLogLength);
        if(infoLogLength > 0)
        {
            log = (char*)malloc(infoLogLength);
            if(log != NULL)
            {
                GLsizei written;
                glGetProgramInfoLog(shader_program_object,infoLogLength,&written,log);
                fopen_s(&gpFile, "Log.txt", "a");
                fprintf(gpFile,"Shader Program Log : %s\n",log);
                fclose(gpFile);
                free(log);
                uninitialize();
            }
        }
    }

    // post linking process

    mvpUniformMatrix = glGetUniformLocation(shader_program_object,"u_mvpMatrix");

    // Initialize OpenCL

    // get platform id

    char chBuffer[1024];

    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;
    cpPlatform;

    ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS || num_platforms <= 0)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "Failed to Find OpenCL Platforms\n");
        fclose(gpFile);
        return -5;
    }
    else
    {
        if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
        {
            fopen_s(&gpFile, "Log.txt", "a");
            fprintf(gpFile, "Failed to allocate memory for cl_platform ID's!\n\n");
            fclose(gpFile);
            return -8;
        }

        ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
        for (cl_uint i = 0; i < num_platforms; ++i)
        {
            ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
            if (ciErrNum == CL_SUCCESS)
            {
                if (strstr(chBuffer, "NVIDIA") != NULL)
                {
                    cpPlatform = clPlatformIDs[i];
                    break;
                }
            }
        }

        if (cpPlatform == NULL)
        {
            fopen_s(&gpFile, "Log.txt", "a");
            fprintf(gpFile, "NVIDIA Platform Not Found\n\n");
            fclose(gpFile);
            cpPlatform = clPlatformIDs[0];
        }
        free(clPlatformIDs);
    }

    // Get The Number Of GPU devices Available to the platform

    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "clGetDeviceIDs failed\n");
        fclose(gpFile);
        return -7;
    }

    cdDevices = new cl_device_id[uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, cdDevices, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "clGetDeviceIDs failed\n");
        fclose(gpFile);
        return -7;
    }

    cl_context_properties props[] =
    {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM,(cl_context_properties)cpPlatform,
        0
    };

    cxGPUcontext = clCreateContext(props, 1, &cdDevices[0], NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "clCreateContext failed\n");
        fclose(gpFile);
        return -7;
    }

    clGetDeviceInfo(cdDevices[0], CL_DEVICE_NAME, sizeof(chBuffer), &chBuffer, NULL);
    fopen_s(&gpFile, "Log.txt", "a");
    fprintf(gpFile, "Selected GPU = %s\n", chBuffer);
    fclose(gpFile);

    // Create Command Queue

    cqCommandQueue = clCreateCommandQueue(cxGPUcontext, cdDevices[0], 0, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "clCreateCommandQueue failed\n");
        fclose(gpFile);
        return -7;
    }

    // OpenCL Program Setup

    size_t program_length;
    std::ifstream kernelFile("simpleGL.cl", std::ios::in);
    if (!kernelFile.is_open())
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "Failed To Open Kernel File\n");
        fclose(gpFile);
        return -7;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();

    cSrcCL = srcStdStr.c_str();

    cpProgram = clCreateProgramWithSource(cxGPUcontext, 1, (const char**)&cSrcCL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "clCreateProgram With Source failed\n");
        fclose(gpFile);
        return -7;
    }

    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(cpProgram, cdDevices[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "Error In Kernel :: %s\n", buildLog);
        fclose(gpFile);
        clReleaseProgram(cpProgram);
        return -7;
    }

    // Create OpenCL Kernel

    ckKernel = clCreateKernel(cpProgram, "sine_wave", &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "clCreateKernel With Source failed\n");
        fclose(gpFile);
        return -7;
    }


    // create VBO
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo_vertex);
    glGenBuffers(1, &vbo_cpu);
    if (onGPU)
    {
        CreateVBO();
    }
    else
    { 
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cpu);
        glBufferData(GL_ARRAY_BUFFER, mesh_width * mesh_height * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(ATTRIBUTE_POSITION);
    }
    // set kernel args
    ciErrNum = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&vbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &mesh_width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &mesh_height);

    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "Error Setting Kernel Arguments.\n");
        fclose(gpFile);
        return -8;
    }

    // run opencl kernel once for generating vertex positions.

    if(onGPU)
        runKernel();
    else
        runCPU();

    //clear screen using blue color:
    glClearColor(0.0f,0.0f,0.0f,1.0f);

    // Depth Related Changes
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    //warmup resize call
    resize(WIN_WIDTH,WIN_HEIGHT);
    return 0;
}

void CreateVBO()
{
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    cl_int err;
    glBindVertexArray(vao);
      glBindBuffer(GL_ARRAY_BUFFER, vbo_vertex);
      glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
      glVertexAttribPointer(ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
      glEnableVertexAttribArray(ATTRIBUTE_POSITION);
      vbo_cl = clCreateFromGLBuffer(cxGPUcontext,CL_MEM_WRITE_ONLY,vbo_vertex,&err);
       if (err != CL_SUCCESS)
       {
            fopen_s(&gpFile, "Log.txt", "a");
            fprintf(gpFile, "Error Creating cl buffer from GL error = %d \n",err);
            fclose(gpFile);
       }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void runCPU()
{
    float freq = 4.0f;
    for (int y = 0; y < mesh_height; y++)
    {
        for (int x = 0; x < mesh_width; x++)
        {
            int offset = x + y * mesh_width;
            float u = (GLfloat)x / (GLfloat)mesh_width;
            float v = (GLfloat)y / (GLfloat)mesh_height;
            float w = sinf(u * freq + angle_triangle) * cosf(v * freq + angle_triangle) * 0.5f;
            
            u = u * 2.0f - 1.0f;
            v = v * 3.0f - 1.0f;

            buffer[offset * 4 + 0] = (GLfloat)u;
            buffer[offset * 4 + 1] = (GLfloat)w;
            buffer[offset * 4 + 2] = (GLfloat)v;
            buffer[offset * 4 + 3] = (GLfloat)1.0f;
        }
    }

    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cpu);
    glBufferData(GL_ARRAY_BUFFER, size, buffer, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void runKernel()
{
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);

    ciErrNum = CL_SUCCESS;
    clFinish(cqCommandQueue);

    ciErrNum = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&vbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &mesh_width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &mesh_height);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &angle_triangle);

    ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue,1,&vbo_cl,0,0,0);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "clEnqueueAcquireGLObjects Failed\n");
        fclose(gpFile);
    }
    /*    ciErrNum = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&vbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &mesh_width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &mesh_height);
    ciErrNum |= clSetKernelArg(ckKernel,3,sizeof(float),&angle_triangle);
    */

    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel,2,NULL,szGlobalWorkSize,NULL,0,0,0);

    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "Set Kernel Args or NDKernelRange Failed\n");
        fclose(gpFile);
    }

    ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl, 0, 0, 0);
    if (ciErrNum != CL_SUCCESS)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile, "Failed To Release OpenGL Object\n");
        fclose(gpFile);
    }
    clFinish(cqCommandQueue);
}

void printGLInfo(void)
{
    // Local variable declarations
    GLint numExtentions = 0;

    // code
    fprintf(gpFile,"OpenGL Vendor : %s\n",glGetString(GL_VENDOR));
    fprintf(gpFile,"OpenGL Renderer : %s\n",glGetString(GL_RENDERER));
    fprintf(gpFile,"OpenGL Version : %s\n",glGetString(GL_VERSION));
    fprintf(gpFile,"OpenGL GLSL Version : %s\n",glGetString(GL_SHADING_LANGUAGE_VERSION));
    glGetIntegerv(GL_NUM_EXTENSIONS,&numExtentions);
    fprintf(gpFile,"Number Of Supported Extensions : %d\n",numExtentions);
    //fprintf(gpFile,"OpenGL  : %s\n",glGetString(GL_VERSION));    
    for (int i = 0; i < numExtentions; i++)
    {
        /* code */
        fprintf(gpFile,"%s\n",glGetStringi(GL_EXTENSIONS,i));
    }
    
}

void resize(int width,int height)
{
    // Code

    GLfloat aspectRatio = 0.0f;

    // to avoid divide by 0 error later in codebase.
    if(height == 0)
        height = 1;
    
    glViewport(0,0,(GLsizei)width,(GLsizei)height);
    perspectiveProjectionMatrix = vmath::perspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);
}

void ComputeFPS()
{

}

void display(void)
{
    
    if (onGPU)
        runKernel();
    else
        runCPU();

    mat4 modelViewMatrix = mat4::identity();
    mat4 translationMatrix = mat4::identity();
    mat4 scaleMatrix = mat4::identity();
    mat4 modelViewProjectionMatrix = mat4::identity();

    // Code
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    translationMatrix = vmath::translate(0.0f,0.0f,-2.0f);
    modelViewMatrix = translationMatrix;
    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

    glPointSize(1.0f);
    // Use The Shader Program object
    glUseProgram(shader_program_object);
    glUniformMatrix4fv(mvpUniformMatrix,1,GL_FALSE,modelViewProjectionMatrix);
    {
        glUniform4f(glGetUniformLocation(shader_program_object, "out_color"), 1.0f, 0.0f, 0.0f, 1.0f);
        glBindVertexArray(vao);
        if (onGPU)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_vertex);
        else
            glBindBuffer(GL_ARRAY_BUFFER, vbo_cpu);

             glDrawArrays(GL_POINTS,0,mesh_width*mesh_height);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    // UnUse The Shader Program Object
    glUseProgram(0);

    SwapBuffers(ghdc);
}

void update(void)
{
    // Code
    if(angle_triangle >= 360.0f)
        angle_triangle = 0.0f;

    angle_triangle += 0.01f;
}

void uninitialize(void)
{
    // Function Declarations

    void ToogleFullScreen(void);

    // Code

    if(gbFullScreen)
    {
        ToogleFullScreen();
    }

    // Shader Uninitialization

    if (ckKernel)    clReleaseKernel(ckKernel);
    if (cpProgram)    clReleaseProgram(cpProgram);
    if (cqCommandQueue)    clReleaseCommandQueue(cqCommandQueue);

	if (vbo_vertex)
	{
		glDeleteBuffers(1, &vbo_vertex);
		vbo_vertex = 0;
	}

    if (vbo_cl)  clReleaseMemObject(vbo_cl);
    if (cxGPUcontext) clReleaseContext(cxGPUcontext);
    if (cSrcCL)  free((void*)cSrcCL);
    if (cdDevices)   delete(cdDevices);

	if (vao)
	{
		glDisableVertexArrayAttrib(vao, ATTRIBUTE_POSITION);
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

    if(shader_program_object)
    {
        glUseProgram(shader_program_object);

        GLsizei num_attached_shaders;
        glGetProgramiv(shader_program_object,GL_ATTACHED_SHADERS,&num_attached_shaders);
        GLuint * shader_objects = NULL;

        shader_objects = (GLuint*)malloc(num_attached_shaders);

        glGetAttachedShaders(shader_program_object,num_attached_shaders,&num_attached_shaders,shader_objects);

        for(GLsizei i = 0; i < num_attached_shaders; i++)
        {
            glDetachShader(shader_program_object,shader_objects[i]);
            glDeleteShader(shader_objects[i]);
            shader_objects[i] = 0;
        }

        free(shader_objects);

        glUseProgram(0);

        glDeleteProgram(shader_program_object);
    }

    if(wglGetCurrentContext() == ghrc)
    {
        wglMakeCurrent(NULL,NULL);
    }

    if(ghrc)
    {
        wglDeleteContext(ghrc);
        ghrc = NULL;
    }

    if(ghdc)
    {
        ReleaseDC(ghwnd,ghdc);
    }

    if(ghwnd)
    {
        DestroyWindow(ghwnd);
    }

    if(gpFile)
    {
        fopen_s(&gpFile, "Log.txt", "a");
        fprintf(gpFile,"Closing Log File.\n");
        fclose(gpFile);
        gpFile = NULL;
    }
}
