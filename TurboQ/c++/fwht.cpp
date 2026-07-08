#include <vector>
#include <iostream>
#include <array>
#include <algorithm>
using namespace std;

class FWHT{
public:
    void foo(vector<float>& arr, int l, int r){
        if(l==r){
            return;
        }
        int mid = (l+r+1)/2;
        foo(arr, l, mid-1); 
        foo(arr, mid, r); 
        for(int i = 0; i < mid-l; i++){
            int p = arr[l+i], q = arr[mid+i];
            arr[l+i] = p + q;
            arr[mid+i] = p-q;
        }
    }
};

int main(){
    int nf = 8;
    int i = 0;
    
    vector<float> random_vector(nf,0);    
    
    transform(random_vector.begin(), random_vector.end(), random_vector.begin(), [i](float n) mutable{
        return n+i++;
    });

    FWHT fwht;
    fwht.foo(random_vector, 0, random_vector.size()-1);
    for(int i : random_vector) cout << i << " ";
    cout << "\n";

    fwht.foo(random_vector, 0, random_vector.size()-1);
    for(float& x: random_vector) x /= nf;
    for(int i : random_vector) cout << i << " ";
    return 1;
}
