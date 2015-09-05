#pragma once
#include <io.h>
#include <vector>
#include <string>
using namespace std;

// 获取path路径下的所有文件名到数组files中
// 参数：输入string path ， 输出 vector<string>& files
void getFiles(string path, vector<string>& files){  

    //文件句柄  
    long   hFile   =   0;  

    //文件信息  
    struct _finddata_t fileinfo;  
    string p;  
    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
    {
        do{  
            //如果是目录,迭代之  
            //如果不是,加入列表  
            if((fileinfo.attrib &  _A_SUBDIR)){  
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
                    getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
            }else{
                files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
            }  
        }
        while(_findnext(hFile, &fileinfo)  == 0);  
            _findclose(hFile);
    }  
}

/* example

char * filePath = "D:\\sample";  
vector<string> files;  
int size = files.size();  
for (int i = 0;i < size;i++)  
{  
    cout<<files[i].c_str()<<endl;  
}

*/