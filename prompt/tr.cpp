#include <bits/stdc++.h>
using namespace std;
const int N = 100005;
struct node {
    string str1, str2;
} a[N];
signed main() {
    //rewrite the code from line 8 to line 32 using ifstream
    ifstream fin("animal-ori.txt");
    string str;
    int cnt = 0;
    while (getline(fin, str)) {
        int pos = str.find("+");
        a[++cnt].str1 = str.substr(0, pos);
        a[cnt].str2 = str.substr(pos + 1);
    }
    fin.close();
    ifstream ffin("daily-ori.txt");
    while (getline(ffin, str)) {
        int pos = str.find("+");
        a[++cnt].str1 = str.substr(0, pos);
        a[cnt].str2 = str.substr(pos + 1);
    }
    ffin.close();
    ifstream fffin("trans-ori.txt");
    while (getline(fffin, str)) {
        int pos = str.find("+");
        a[++cnt].str1 = str.substr(0, pos);
        a[cnt].str2 = str.substr(pos + 1);
    }
    fffin.close();


    // search all the str2 in a[] and replace the space with "-". If there is no alphabet letter after the space, it will be ignored.
    // cout << "cnt = " << cnt << '\n';
    for (int i = 1; i <= cnt; i++) {
        int pos = a[i].str2.find(" ");
        if (pos != -1) {
            if (pos + 1 < a[i].str2.length() && isalpha(a[i].str2[pos + 1])) {
                a[i].str2[pos] = '-';
            } else {
                a[i].str2 = a[i].str2.substr(0, pos);
            }
        }
    }
    // put all the str1 into front.txt and all the str2 into back.txt
    freopen("front.txt", "w", stdout);
    for (int i = 1; i <= cnt; i++) {
        cout << a[i].str1 << endl;
    }
    freopen("back.txt", "w", stdout);
    for (int i = 1; i <= cnt; i++) {
        cout << a[i].str2 << endl;
    }
    return 0;
}