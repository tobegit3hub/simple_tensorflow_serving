#include<stdio.h>
#include<iostream>
#include<string>
#include<vector>

using namespace std;

void print_array(float a[]) {
    for (float*p = a; *p != 0; p++) {
        cout<<*p<<endl;
    }
}

int main() {
    int a=5;
    printf("hellow from c style \n");

    std::cout<<"standard hello man"<<std::endl;

    cout<<"cout without std namespace"<<endl;

    char c = 'a';
    char* p = &c;
    cout<< *p <<endl;

    float array[5] = {1, 2, 3, 4, 5};
    vector<float> brryay(10);
    print_array(array);













}


void some_function() {
  double a = 2 * 2;
  int i = 7;
  int l = i + 7;

}

class complex {

public:
  int my_function() {
    return 10;
  }

};

struct Person {
  string name;
  int age;
};
