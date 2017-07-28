#include "include/get_dm.hpp"
#include "include/get_nel.hpp"
#include "include/get_eel.hpp"
#include "include/read_in_files.hpp"

#include <iostream>
#include <vector>

using namespace std;

int main()
{
    int nbas = 7, nocc = 5;
    vector<double> H = read_matrix_file("H.data", nbas, nbas);
    vector<double> F = read_matrix_file("F.data", nbas, nbas);
    vector<double> S = read_matrix_file("S.data", nbas, nbas);
    vector<double> C = read_matrix_file("C.data", nbas, nbas);

    vector<double> D = get_dm(C, nbas, nocc);

    int nel = get_nel(D, S);

    double E = get_eel(H, F, D, nbas);

    cout << "nel = " << nel << endl;
    cout << "etot = " << E + 8.0023664507190784 << endl;

    return 0;
}
